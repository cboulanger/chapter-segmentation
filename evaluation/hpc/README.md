# Running the NuExtract-2.0-4B fine-tuning pilot on MPCDF HPC

This package offloads the fine-tuning pilot described in
`docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md`
(and documented for local runs in `evaluation/README.md`'s "NuExtract-2.0-4B
fine-tuning pilot" section) onto MPCDF's SLURM/Apptainer HPC systems
(Raven), instead of running it on a local machine. It reuses the exact
same scripts (`prepare_nuextract_finetune_data.py`, `finetune_nuextract.py`,
`merge_nuextract_lora.py`, `evaluate_nuextract_finetune.py`) unmodified --
only the container and orchestration are new.

Blueprints for this were adapted from
[`ai_containers`](https://gitlab.mpcdf.mpg.de/dataanalytics-public/ai_containers)
and
[`llms-meet-mpcdf`](https://gitlab.mpcdf.mpg.de/dataanalytics-public/llms-meet-mpcdf)
(MPCDF's own example repos). Neither had a ready-made single-GPU LoRA
example -- `llms-meet-mpcdf`'s closest match, `sft_with_fsdp`, is an
8-node/32-GPU full fine-tune of a 70B model, which is unnecessary here: a
4B-param LoRA adapter trained on ~39 examples fits comfortably on one GPU.

Files in this directory:

- `nuextract.def` -- Apptainer definition: NVIDIA's PyTorch base image +
  `transformers`/`peft`/`accelerate`/`huggingface-hub` (same versions as
  `pyproject.toml`'s `nuextract-finetune` optional-dependency group) + a
  CUDA-enabled `llama-cpp-python` + a from-source `llama.cpp` build (for
  `convert_hf_to_gguf.py` and `llama-quantize`, neither of which ships as
  a pip package).
- `run_pilot.slurm` -- a single batch job running all five pipeline steps
  in order: prepare data, train, merge, convert+quantize, score (both the
  fine-tuned and base checkpoints, on the same held-out split).

Nothing under this directory needs `--nv`/GPU access except the training
and scoring steps -- data prep and GGUF conversion/quantization are CPU
work and run without it, so they don't waste GPU allocation time.

## 0. Data governance note

Since this HPC node is private, PDFs and ground truth for
`copyrighted-scans/` books can be uploaded directly (see step 2) rather
than only shipping pre-extracted text -- there's no need to run
`prepare_nuextract_finetune_data.py` locally first and ship just the
JSONL, the way you'd want to on a shared/less-trusted machine. The
existing `evaluation/.gitignore` rules (`*.pdf`, `manifest.local.json`,
`finetune/`) still apply here: none of that data is ever committed to
git, on either machine.

## 1. Clone the branch on the HPC login node

```bash
git clone -b worktree-nuextract-baseline-spike \
    https://github.com/cboulanger/chapter-segmentation.git
cd chapter-segmentation
```

(Or `git pull` if you already have a checkout there.)

## 2. Upload the evaluation corpus

From your local machine, copy the PDFs and any local-only manifest files
into the matching `evaluation/corpus/<corpus>/` directory on the HPC
side (adjust paths/host to your actual setup):

```bash
rsync -av --include="*.pdf" --include="manifest.local.json" --include="*/" --exclude="*" \
    evaluation/corpus/ \
    hpc:/path/to/chapter-segmentation/evaluation/corpus/
```

Books already in the committed `manifest.json` only need their PDF
uploaded (the ground-truth `.expected.json` is already in the git
checkout); books that only exist in a local `manifest.local.json` need
both. See `evaluation/CLAUDE.md` if you're unsure which category a given
book falls into.

## 3. Build the container (login node)

```bash
module load apptainer/1.4.3
cd evaluation/hpc
apptainer build --fakeroot nuextract.sif nuextract.def
```

This compiles `llama.cpp` from source and installs the ML stack, so
expect it to take a while (comparable to the multi-GB base-container
pulls other MPCDF examples warn about). `--fakeroot` is required here
(unlike most other blueprints in `ai_containers`) because this container
also runs `apt-get install` and `cmake --build` in `%post`.

Sanity-check the build:

```bash
apptainer test nuextract.sif
```

Expected: `python deps ok`, `llama-quantize built ok`,
`convert_hf_to_gguf.py present ok`.

## 4. Pre-fetch model weights (login node -- compute nodes have no internet)

Compute nodes on MPCDF systems cannot reach the internet (the same
reason `llms-meet-mpcdf/sft_with_fsdp`'s README documents WandB's offline
mode). Download `numind/NuExtract-2.0-4B` and the published base-model
GGUF into `$HF_HOME` from the login node *before* submitting the job:

```bash
export HF_HOME=/ptmp/$USER/huggingface
mkdir -p "$HF_HOME"

apptainer exec evaluation/hpc/nuextract.sif \
    huggingface-cli download numind/NuExtract-2.0-4B
apptainer exec evaluation/hpc/nuextract.sif \
    huggingface-cli download numind/NuExtract-2.0-4B-GGUF NuExtract-2.0-4B-Q4_K_M.gguf
```

Both repos are public as of this writing, so no `HF_TOKEN` should be
needed; if you hit a gated-repo error, request access on Hugging Face and
re-run with `HF_TOKEN=<your token>` prefixed.

## 5. Submit the job

```bash
cd /path/to/chapter-segmentation   # repo root, not evaluation/hpc
sbatch evaluation/hpc/run_pilot.slurm
```

`run_pilot.slurm` assumes it's submitted from the repo root and that
`evaluation/hpc/nuextract.sif` already exists (step 3). It runs all five
pipeline steps as one job:

1. `prepare_nuextract_finetune_data.py` (CPU only)
2. `finetune_nuextract.py --device cuda` (GPU)
3. `merge_nuextract_lora.py` (GPU node, but CPU-bound work)
4. `convert_hf_to_gguf.py` + `llama-quantize` (CPU only)
5. `evaluate_nuextract_finetune.py`, twice -- fine-tuned checkpoint, then
   the unmodified base model on the same split (GPU, via `llama.cpp`)

Default `--gres=gpu:a100:1` and `--time=02:00:00` -- generous margins for
a job this size; adjust the GPU type in `run_pilot.slurm` if your account
doesn't have A100 access (check with `sinfo` or the MPCDF docs).

Monitor with `squeue --me` and `tail -f job.out.<jobid>`.

## 6. Retrieve results

The two eval passes' full output is both in `job.out.<jobid>` and saved
separately at `/ptmp/$USER/nuextract-pilot/eval-{finetuned,base}.txt`.
Record both f1 numbers (and whether the null-page-number rate dropped on
held-out books -- inspect a couple by hand) in `evaluation/RESULTS.md`,
per the design spec's "Decision criteria" and the existing pattern for
every prior NuExtract finding in that file.

The adapter/merged/GGUF artifacts stay under `/ptmp/$USER/nuextract-pilot/`
-- copy them back with `scp`/`rsync` only if you want to keep the
fine-tuned weights around; they're not needed to record the pilot's
result.

## A note on the llama.cpp-only eval constraint

The pilot's design mandates scoring only through `llama.cpp`/GGUF, never
`transformers`' own generation -- but that rule was root-caused to an
Apple Silicon **MPS**-specific decoding bug (see
`evaluation/RESULTS.md`'s transformers/MPS-vs-llama.cpp finding). On an
NVIDIA GPU, that specific bug likely doesn't apply, so a `transformers`-
on-CUDA generation pass would be a cheap way to double-check the MPS
finding really was MPS-specific. `run_pilot.slurm` doesn't do this by
default (keeping the recorded number directly comparable to the
already-measured Mac baseline in `RESULTS.md`), but it's a reasonable
follow-up experiment if you want extra confidence in the numbers.
