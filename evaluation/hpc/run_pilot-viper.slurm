#!/bin/bash -l
#
#SBATCH -o job.out.%j
#SBATCH -e job.err.%j
#SBATCH -D ./
#SBATCH -J nuextract-lora-pilot-viper
#
#SBATCH --account=mrg_apu
#SBATCH --partition=apu1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#
#SBATCH --mail-type=none
#
# Sibling of ./run_pilot.slurm (Raven/NVIDIA); kept as a fully separate
# file rather than a parameterized/shared one so either target can be
# edited freely without the other silently breaking. See that file for
# line-by-line comparison and evaluation/hpc/README.md for narrative
# docs (Raven-only as written -- no Viper walkthrough exists yet).
#
# Viper-GPU nodes: 2x AMD Instinct MI300A APUs per node, 128GB HBM3
# each, 220GB usable RAM per node, 96 logical CPUs (SMT) total --
# https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html.
# --gres=gpu:1 requests one APU (shared-node); --cpus-per-task=24 is
# that APU's physical core share, per the guide's own example
# (ntasks-per-node * cpus-per-task must not exceed 96).
#
# --account/--partition are both required, not optional defaults:
# `sacctmgr -p show assoc user=$USER` showed two separate accounts
# (mrg_apu, mrg_cpu) with no default partition, and none of Viper-GPU's
# apu/apu1/apudev partitions are Default=YES (`scontrol show partition`).
# apu1 (not plain apu) is a single-node partition matching this job's
# single-GPU shape, and its own DefCpuPerGPU=48/DefMemPerNode=110000
# happen to match --cpus-per-task/--mem below exactly -- confirms
# --mem=110000 (previously an unverified half-of-220GB guess ported from
# Raven's separate-host-RAM/GPU-VRAM accounting, which doesn't obviously
# apply to MI300A's unified-memory APU design) is a sane value after all.
#
# CRITICAL: sbatch/srun for apu/apu1/apudev must be run from viper11,
# viper12, or viper13 specifically -- `scontrol show partition`'s
# AllocNodes=viper[11-13] for all three restricts which login nodes may
# even submit to them. Submitting from viper01/02 (fine for everything
# else, including building the .sif itself) fails with "Unable to
# allocate resources: Access/permission denied" before Slurm even gets
# to evaluating the resource request itself.
#SBATCH --time=02:00:00

# See evaluation/hpc/README.md's "Running on Viper-GPU instead" for the
# full setup this assumes (adjusted for Viper): repo cloned at $REPO_DIR,
# evaluation/finetune/data/{train,eval}.jsonl already rsynced over from a
# local run of prepare_nuextract_finetune_data.py (this script never
# touches the PDFs themselves), the container built at
# $REPO_DIR/evaluation/hpc/nuextract-viper.sif, and the model weights + base
# GGUF already pre-fetched into $HF_HOME from a login node (compute nodes
#
# IMPORTANT: $REPO_DIR/$PTMP below resolve to whatever /u and /ptmp mean
# on whichever node this job actually runs on. Viper-CPU and Viper-GPU
# have completely separate filesystems at those same paths -- a checkout
# done from viper01/02 is invisible to the vipa* compute nodes this job
# runs on. Everything must already live under Viper-GPU's own /u and
# /ptmp (clone and rsync directly from viper11/12/13) before submitting
# -- see the README section above for the cross-filesystem copy command
# if you set up on Viper-CPU first.
# have no internet).

set -e

module purge
# Confirmed available on Viper via `module avail apptainer` (also offers
# 1.3.2, 1.3.6, 1.4.1, 1.5.2) -- 1.4.3 matches run_pilot.slurm's Raven value.
module load apptainer/1.4.3

REPO_DIR="${REPO_DIR:-$PWD}"
PTMP="/ptmp/${USER}"
SIF="${REPO_DIR}/evaluation/hpc/nuextract-viper.sif"
BIND="${REPO_DIR}:${REPO_DIR},${PTMP}"

export HF_HOME="${PTMP}/huggingface"
export HF_HUB_OFFLINE=1

ADAPTER_DIR="${PTMP}/nuextract-pilot/adapter"
MERGED_DIR="${PTMP}/nuextract-pilot/merged"
FP16_GGUF="${PTMP}/nuextract-pilot/merged.fp16.gguf"
Q4_GGUF="${PTMP}/nuextract-pilot/merged.Q4_K_M.gguf"
mkdir -p "${PTMP}/nuextract-pilot"

cd "$REPO_DIR"

TRAIN_JSONL="${REPO_DIR}/evaluation/finetune/data/train.jsonl"
EVAL_JSONL="${REPO_DIR}/evaluation/finetune/data/eval.jsonl"
if [[ ! -f "$TRAIN_JSONL" || ! -f "$EVAL_JSONL" ]]; then
    echo "Missing $TRAIN_JSONL and/or $EVAL_JSONL." >&2
    echo "This job never runs prepare_nuextract_finetune_data.py itself --" >&2
    echo "run it locally and rsync evaluation/finetune/data/ over first" >&2
    echo "(see evaluation/hpc/README.md's 'Data transfer' section)." >&2
    exit 1
fi

echo "=== [1/5] train LoRA adapter ($(date)) ==="
srun apptainer exec --rocm -B "$BIND" "$SIF" \
    python evaluation/scripts/finetune_nuextract.py \
        --output-dir "$ADAPTER_DIR" --device cuda

echo "=== [2/5] merge adapter into base weights ($(date)) ==="
srun apptainer exec --rocm -B "$BIND" "$SIF" \
    python evaluation/scripts/merge_nuextract_lora.py \
        --adapter-dir "$ADAPTER_DIR" --output-dir "$MERGED_DIR"

echo "=== [3/5] convert + quantize to GGUF ($(date)) ==="
QUANTIZE_BIN=$(srun apptainer exec -B "$BIND" "$SIF" \
    find /opt/llama.cpp/build -name "llama-quantize" -type f | head -n1)
echo "Using llama-quantize at: $QUANTIZE_BIN"

srun apptainer exec -B "$BIND" "$SIF" \
    python /opt/llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" --outfile "$FP16_GGUF"
srun apptainer exec -B "$BIND" "$SIF" \
    "$QUANTIZE_BIN" "$FP16_GGUF" "$Q4_GGUF" Q4_K_M

echo "=== [4/5] score fine-tuned checkpoint on held-out split ($(date)) ==="
srun apptainer exec --rocm -B "$BIND" "$SIF" \
    python evaluation/scripts/evaluate_nuextract_finetune.py --gguf-path "$Q4_GGUF" \
    | tee "${PTMP}/nuextract-pilot/eval-finetuned.txt"

echo "=== [5/5] score base checkpoint on the same split -- the number to beat ($(date)) ==="
srun apptainer exec --rocm -B "$BIND" "$SIF" \
    python evaluation/scripts/evaluate_nuextract_finetune.py \
    | tee "${PTMP}/nuextract-pilot/eval-base.txt"

echo "=== done ($(date)) ==="
echo "Adapter:        $ADAPTER_DIR"
echo "Merged/GGUF:    $MERGED_DIR, $Q4_GGUF"
echo "Eval summaries: ${PTMP}/nuextract-pilot/eval-{finetuned,base}.txt"
