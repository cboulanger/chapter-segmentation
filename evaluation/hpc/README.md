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
4B-param LoRA adapter trained on ~59 examples fits comfortably on one GPU.

Files in this directory:

- `nuextract.def` -- Apptainer definition: NVIDIA's PyTorch base image +
  `transformers`/`peft`/`accelerate`/`huggingface-hub` (same versions as
  `pyproject.toml`'s `nuextract-finetune` optional-dependency group) + a
  CUDA-enabled `llama-cpp-python` + a from-source `llama.cpp` build (for
  `convert_hf_to_gguf.py` and `llama-quantize`, neither of which ships as
  a pip package).
- `run_pilot.slurm` -- a single batch job running the four remaining
  pipeline steps in order: train, merge, convert+quantize, score (both
  the fine-tuned and base checkpoints, on the same held-out split).

Nothing under this directory needs `--nv`/GPU access except the training
and scoring steps -- GGUF conversion/quantization is CPU work and runs
without it, so it doesn't waste GPU allocation time.

## 0. Data transfer: ship the extracted text, not the PDFs

`prepare_nuextract_finetune_data.py` is the *only* pipeline step that
touches the PDFs -- it reads each one just to extract its TOC-scan-window
text (the same `analysis_pages_for` pypdf/OCR-cache path the rest of the
evaluation suite uses), then writes that text plus the target JSON (built
from the already-committed `.expected.json` files) into
`evaluation/finetune/data/{train,eval}.jsonl`. Nothing downstream --
training, merging, GGUF conversion, or scoring -- ever opens a PDF again;
they only read those two JSONL files.

That makes the ~800MB PDF corpus itself the wrong thing to transfer,
independent of whether the HPC node is trusted with copyrighted data:
`train.jsonl`/`eval.jsonl` are orders of magnitude smaller (a few MB, not
hundreds) and are all `run_pilot.slurm` needs. **Run the prepare step
locally, where the PDFs and any OCR cache already sit, and rsync only its
output:**

```bash
uv run python evaluation/scripts/prepare_nuextract_finetune_data.py

rsync -av evaluation/finetune/data/ \
    cboul@raven.mpcdf.mpg.de:/u/cboul/projects/chapter-segmentation/evaluation/finetune/data/
```

This whole package targets **Raven** specifically (NVIDIA/A100 -- see
`nuextract.def`'s CUDA base image and `run_pilot.slurm`'s `--gres=gpu:a100:1`
and `--nv`), not Viper (AMD/ROCm, which needs a different base image,
`-DGGML_HIP=ON` instead of `-DGGML_CUDA=ON`, `--rocm` instead of `--nv`,
and Viper's own `--constraint`/`--gres` syntax -- none of that is done
here). Double-check `raven.mpcdf.mpg.de` above, not `viper.mpcdf.mpg.de`.

(Swap in your actual remote username/host/path -- and note the `:`
immediately after the host: `rsync` treats anything without it as a
second local path, which silently recreates the whole destination as a
nested local directory instead of transferring anywhere, so double-check
this before running it for real. If MPCDF's `$HOME`/project filesystem is
shared across Raven and Viper for your account -- check with `ssh
raven.mpcdf.mpg.de 'ls ~/projects/chapter-segmentation'` -- you may not
need to re-clone/re-transfer anything at all, just log into Raven instead
of Viper and pick up from step 3 below.)

The existing `evaluation/.gitignore` rule (`finetune/`) means this
directory is never committed to git on either machine -- transfer it out
of band, the same way the PDFs themselves already are.

## 1. Clone the branch on the HPC login node

```bash
git clone -b worktree-nuextract-baseline-spike \
    https://github.com/cboulanger/chapter-segmentation.git
cd chapter-segmentation
```

(Or `git pull` if you already have a checkout there.)

## 2. Upload the prepared training/eval data

See step 0 above -- run `prepare_nuextract_finetune_data.py` locally
first, then `rsync` just `evaluation/finetune/data/` to the same path
inside your HPC checkout.

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

> [!note]
> `%post` runs with no GPU attached and, on a login node, a capped
> per-user process count -- `nuextract.def` accounts for both: it pins
> `CMAKE_CUDA_ARCHITECTURES` to `80;90` (A100/H100/H200) instead of
> relying on runtime GPU detection (which finds nothing at build time),
> and caps build parallelism to 4 instead of letting the CUDA build
> spawn one `nvcc` process per core. If the build still fails with
> repeated `fork: retry: Resource temporarily unavailable` (a process/
> `ulimit -u` limit, not a fakeroot problem -- ignore Apptainer's generic
> fakeroot troubleshooting hint in that case), the login node is likely
> under load from other users; build inside a short job on a compute
> node instead:
>
> ```bash
> srun --time=00:45:00 --cpus-per-task=8 --mem=32G \
>     apptainer build --fakeroot nuextract.sif nuextract.def
> ```

Sanity-check the build:

```bash
apptainer test nuextract.sif
```

Expected: `python deps ok (excl. llama_cpp ...)`, `llama_cpp shared
library built ok`, `llama-quantize built ok`, `convert_hf_to_gguf.py
present ok`. `llama_cpp` itself is deliberately not imported by this
test -- its compiled library needs a real GPU driver (`--nv`), not
available at build time. Do that check once for real, on a GPU node:

```bash
srun --gres=gpu:a100:1 --cpus-per-task=18 --mem=125000 --constraint="gpu" \
    --time=00:05:00 --pty bash
```

(`--mem` must be a proportional share of the node's 512GB when you
request fewer than all 4 GPUs -- see `run_pilot.slurm`'s comment on this;
omitting it, or passing `--mem=0`, fails with "requested only 1 of four
gpus but more than 1/4 of memory of the node".) Then, inside that
session:

```bash
apptainer exec --nv nuextract.sif python3 -c "import llama_cpp; print('llama_cpp CUDA import ok')"
```

## 4. Pre-fetch model weights (login node -- compute nodes have no internet)

Compute nodes on MPCDF systems cannot reach the internet (the same
reason `llms-meet-mpcdf/sft_with_fsdp`'s README documents WandB's offline
mode). Download `numind/NuExtract-2.0-4B` and the published base-model
GGUF into `$HF_HOME` from the login node *before* submitting the job:

```bash
export HF_HOME=/ptmp/$USER/huggingface
export HF_TOKEN=<your hugging face access token>
mkdir -p "$HF_HOME"

cd evaluation/hpc
apptainer exec -B /ptmp/$USER nuextract.sif hf download numind/NuExtract-2.0-4B
apptainer exec -B /ptmp/$USER nuextract.sif hf download numind/NuExtract-2.0-4B-GGUF NuExtract-2.0-4B-Q4_K_M.gguf
```

`-B /ptmp/$USER` is required, not optional -- Apptainer's default bind
set doesn't include `/ptmp`, so without it `/ptmp` is visible inside the
container but read-only, and `hf download` fails trying to write its
cache there (`OSError: [Errno 30] Read-only file system: '/ptmp'`). This
matches every other MPCDF example that writes to `/ptmp` (e.g.
`llms-meet-mpcdf/sft_with_fsdp`'s download step) -- they all bind it
explicitly rather than relying on it being writable by default.

## 5. Submit the job

```bash
cd /path/to/chapter-segmentation   # repo root, not evaluation/hpc
sbatch evaluation/hpc/run_pilot.slurm
```

`run_pilot.slurm` assumes it's submitted from the repo root, that
`evaluation/hpc/nuextract.sif` already exists (step 3), and that
`evaluation/finetune/data/{train,eval}.jsonl` are already present (step
2 -- it fails fast with a clear message if they're missing). It runs the
remaining four pipeline steps as one job:

1. `finetune_nuextract.py --device cuda` (GPU)
2. `merge_nuextract_lora.py` (GPU node, but CPU-bound work)
3. `convert_hf_to_gguf.py` + `llama-quantize` (CPU only)
4. `evaluate_nuextract_finetune.py`, twice -- fine-tuned checkpoint, then
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
