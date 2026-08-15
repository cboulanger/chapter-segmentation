#!/bin/bash -l
#
#SBATCH -o job.out.%j
#SBATCH -e job.err.%j
#SBATCH -D ./
#SBATCH -J nuextract-lora-pilot
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#
# A single A100 is comfortably enough for a 4B-param LoRA job on ~78
# training examples -- no multi-node/torchrun complexity needed (unlike
# every multi-GPU example in ai_containers/llms-meet-mpcdf). Raven's GPU
# nodes have 4x A100 (40GB each) + 512GB RAM total; requesting 1 of 4
# GPUs caps you to a proportional share of the node (--cpus-per-task=18
# = 72/4, --mem=125000 = ~512000/4 minus overhead) -- per
# https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html.
# Asking for more memory than your GPU share entitles you to (e.g.
# --mem=0, which means "the whole node") fails with "requested only 1
# of four gpus but more than 1/4 of memory of the node". If your account
# only has access to a different GPU type (H100, RTX6000, ...), change
# --gres above and re-derive --cpus-per-task/--mem the same way -- see
# `sinfo` or the MPCDF docs for what's available under your account/
# partition.
#SBATCH --time=02:00:00

# See evaluation/hpc/README.md for the full setup this script assumes:
# repo cloned at $REPO_DIR, evaluation/finetune/data/{train,eval}.jsonl
# already rsynced over from a local run of prepare_nuextract_finetune_data.py
# (this script never touches the PDFs themselves -- see the README's "Data
# transfer" section for why), the container built at
# $REPO_DIR/evaluation/hpc/nuextract.sif, and the model weights + base
# GGUF already pre-fetched into $HF_HOME from a login node (compute nodes
# have no internet -- see llms-meet-mpcdf/sft_with_fsdp/README.md).

set -e

module purge
module load apptainer/1.4.3

REPO_DIR="${REPO_DIR:-$PWD}"
PTMP="/ptmp/${USER}"
SIF="${REPO_DIR}/evaluation/hpc/nuextract.sif"
BIND="${REPO_DIR}:${REPO_DIR},${PTMP}"

export HF_HOME="${PTMP}/huggingface"
export HF_HUB_OFFLINE=1
# The pipeline scripts each `sys.path.insert` their own repo root, which
# makes `evaluation.*` importable but NOT `chapter_segmentation.*` --
# that package actually lives at src/chapter_segmentation, a different
# directory none of the scripts add to sys.path themselves. Locally this
# is masked by .venv's editable install of the project; the container
# never gets one (deliberately -- it only bind-mounts the repo, see the
# "Data transfer" note above), so it needs this instead. Apptainer
# passes the host environment through by default (no --cleanenv used
# anywhere below), so this reaches every `apptainer exec ... python`
# call that follows.
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}"
# Free mitigation against allocator fragmentation, straight from the CUDA
# OOM error's own suggested fix -- doesn't replace the real fix
# (gradient checkpointing, in finetune_nuextract.py) but costs nothing
# to also set. Must be set before the first CUDA allocation, so exported
# here rather than inside the script.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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
srun apptainer exec --nv -B "$BIND" "$SIF" \
    python evaluation/scripts/finetune_nuextract.py \
        --output-dir "$ADAPTER_DIR" --device cuda

echo "=== [2/5] merge adapter into base weights ($(date)) ==="
srun apptainer exec --nv -B "$BIND" "$SIF" \
    python evaluation/scripts/merge_nuextract_lora.py \
        --adapter-dir "$ADAPTER_DIR" --output-dir "$MERGED_DIR"

echo "=== [3/5] convert + quantize to GGUF ($(date)) ==="
QUANTIZE_BIN=$(srun apptainer exec -B "$BIND" "$SIF" \
    find /opt/llama.cpp/build -name "llama-quantize" -type f | head -n1)
echo "Using llama-quantize at: $QUANTIZE_BIN"

srun apptainer exec -B "$BIND" "$SIF" \
    python /opt/llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" --outfile "$FP16_GGUF"
# --nv IS needed here, unlike convert_hf_to_gguf.py above -- nuextract.def
# builds the whole llama.cpp tree with -DGGML_CUDA=ON, so llama-quantize
# is dynamically linked against libcuda.so.1 even though quantization
# itself never executes GPU code. Without --nv (bind-mounting the host's
# CUDA driver libs), the binary fails to even load: "error while loading
# shared libraries: libcuda.so.1: cannot open shared object file". Costs
# nothing extra -- the whole job already reserves 1 GPU throughout via
# #SBATCH --gres=gpu:a100:1, this doesn't request anything additional.
srun apptainer exec --nv -B "$BIND" "$SIF" \
    "$QUANTIZE_BIN" "$FP16_GGUF" "$Q4_GGUF" Q4_K_M

echo "=== [4/5] score fine-tuned checkpoint on held-out split ($(date)) ==="
srun apptainer exec --nv -B "$BIND" "$SIF" \
    python evaluation/scripts/evaluate_nuextract_finetune.py --gguf-path "$Q4_GGUF" \
        --dump-dir "${PTMP}/nuextract-pilot/predictions-finetuned" \
    | tee "${PTMP}/nuextract-pilot/eval-finetuned.txt"

echo "=== [5/5] score base checkpoint on the same split -- the number to beat ($(date)) ==="
srun apptainer exec --nv -B "$BIND" "$SIF" \
    python evaluation/scripts/evaluate_nuextract_finetune.py \
        --dump-dir "${PTMP}/nuextract-pilot/predictions-base" \
    | tee "${PTMP}/nuextract-pilot/eval-base.txt"

echo "=== done ($(date)) ==="
echo "Adapter:        $ADAPTER_DIR"
echo "Merged/GGUF:    $MERGED_DIR, $Q4_GGUF"
echo "Eval summaries: ${PTMP}/nuextract-pilot/eval-{finetuned,base}.txt"
echo "Raw per-book predictions: ${PTMP}/nuextract-pilot/predictions-{finetuned,base}/"
