# NuExtract-2.0-4B TOC-extraction fine-tuning pilot

**This is a snapshot of an experimental fine-tuning pilot, not a result of
the main chapter-segmentation workflow.** This investigates whether a
small, locally-runnable extraction model (`numind/NuExtract-2.0-4B`,
targeting a no-GPU, 16GB RAM Linux deployment) can extract a book's table
of contents directly, as a candidate alternative/supplement to the
LLM-strategy row in `evaluation/RESULTS.md`'s "Per-strategy standalone
results". It has not been integrated into `chapter_segmentation`. "Current
status" below is expected to go stale and be rewritten whenever the pilot
is re-run or changed; "History" holds the full write-up for every
superseded run and rejected model along the way, kept in full so the
reasoning and dead ends behind the current numbers aren't lost.

## Current status

### NuExtract-2.0-4B zero-shot baseline (2026-08-10)

#### Backend-dependent bug: `transformers`/MPS silently drops `printed_page_number`

Before trusting a full-corpus number for NuExtract-2.0-4B, a zero-shot run
was attempted via the same `transformers`+`mps`+fp16 path used earlier for
NuExtract3, for consistency. It scored **precision=0.12 recall=0.12
f1=0.12** -- far below the model's own 5-book CPU/`llama.cpp` sample
(f1=0.97) using the *same* five books. Investigation (single-book replay,
`evaluation/corpus/open-access/9781771993661.pdf`) found the root cause:
on `transformers`/MPS, at both fp16 and bf16, the model reliably emits
`"printed_page_number": null` for every chapter entry, even though the
scanned page text plainly contains the printed page numbers (verified by
printing the raw scan window) and the model correctly extracts every
title/author. `match_toc_entries` (`evaluation/nuextract_baseline.py`)
requires a non-null page-number match, so this alone drives recall to
near zero regardless of title-extraction quality. Ruled out tokenization
as the cause (`add_special_tokens=True` vs `False` produced identical
token IDs and identical -- still-null -- output). The same prompt run
through `llama.cpp` (GGUF Q4_K_M, both CPU-only and Metal-offloaded)
correctly filled in every page number and scored f1=0.95 on that book,
reproduced twice. This is a genuine backend-dependent decoding
difference for this model, not a fluke, a tokenization bug, or noise --
likely something in how `llama.cpp`'s own prompt tokenization or KV/RoPE
handling differs subtly from the `transformers` path for this specific
architecture. Since the deployment target is `llama.cpp` on a no-GPU
Linux host anyway, this is moot for production, but it means **any
zero-shot/fine-tuning comparison must go through `llama.cpp`, not
`transformers`/MPS** -- the two backends are not interchangeable for this
model's structured-output behavior.

#### Next: fine-tuning feasibility

Before investing in a bigger ground-truth set, the plan is to check
whether LoRA fine-tuning actually moves this number, using a held-out
split of the existing 50-book corpus as a cheap pilot rather than
committing to more ground-truth curation first. Design and implementation
plan: `docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md`.

#### Output-token-limit retest (2026-08-10)

Before the fine-tuning pilot above, cleared the cheaper explanation
first: the failure-mode breakdown's truncation cluster (10 books, 20%)
used `max_tokens=1500`, and several of those books' generation times
(200-500s) are consistent with hitting that cap rather than reaching a
natural stop. Re-ran the full 50-book corpus with `max_tokens=6000`
(everything else identical: `llama.cpp`, Metal-offloaded, `n_ctx=40960`)
to see how much of the f1=0.39 baseline was an artifact of an
under-provisioned output budget rather than a genuine capability gap.

| Corpus | precision | recall | F1 (1500 tok) | F1 (6000 tok) |
| --- | --- | --- | --- | --- |
| copyrighted-scans | 0.57 | 0.42 | 0.17 | **0.48** |
| open-access | 0.39 | 0.46 | 0.47 | 0.42 |
| **Total** | **0.43** | **0.45** | **0.39** | **0.44** |

**A real but partial fix: f1=0.39 -> f1=0.44, driven by 2 of the 10
originally-truncated books recovering completely, not by the cluster
closing.** Tracing all 10 originally-flagged books individually:

- **2 books recovered to strong scores**, confirming the budget really
  was the cause for these: `9783428042241.pdf` (0/0 found -> 38/41
  found, f1 0.00 -> 0.94) and `9783899496291.pdf` (0/0 -> 53/58 found,
  f1 0.00 -> 0.91). Both are large Festschrift-style
  `copyrighted-scans` books with big TOCs -- exactly the shape of book
  the 1500-token cap was too small for -- and both now score
  essentially as well as the corpus's best books, which is why
  `copyrighted-scans`' aggregate f1 nearly tripled (0.17 -> 0.48).
- **4 books still hit the new, 4x-larger cap and still produce zero**:
  `dnb-36942798X.pdf`, `9783839458013.pdf`, `9781783742806.pdf`,
  `9781783743339.pdf` (all logged `[HIT_MAX_TOKENS]`, taking 220-630s
  each). These have TOCs large/complex enough that even 6000 tokens
  isn't enough, or the model enters a repetition loop that never
  reaches valid JSON regardless of budget -- not distinguished further
  here.
- **1 book (`9783848704316.pdf`) took 979s and still produced 0/0
  without hitting the cap** -- it stopped generating on its own before
  6000 tokens, just never produced valid/matching JSON. A different
  failure shape than budget exhaustion.
- **3 books stopped truncating but now produce wrong content instead of
  no content** -- `9782375460122.pdf` (0/0 -> 0/78 found, still 0 true
  positives -- this is the already-documented French-language/
  cataloging-page miss, see the NuExtract3 section above, not a
  truncation artifact at all), `9783839446270.pdf` (0/0 -> 0/0 found in
  29.5s, a fast empty result, not budget-related), and
  `9783839465776.pdf` (0/0 -> 0/59 found, still 0 true positives). Where
  more output budget just means more room to generate spurious entries,
  it doesn't help.

So raising the token budget is a real, worthwhile, free fix -- worth
keeping in whatever configuration the fine-tuning pilot's evaluation
script uses (`evaluate_nuextract_finetune.py` already defaults to
`--max-tokens 6000`) -- but it only fully resolved 2 of 10 originally-
truncated books; the rest were already, or became once budget was no
longer the bottleneck, cases of the model producing wrong or repetitive
output rather than running out of room. **f1=0.44 (this retest), not
f1=0.39, is the correct "baseline to beat" for the fine-tuning pilot**
per its design spec's decision criteria.

**Follow-up: raising the cap further (12000 tokens) does not rescue the
remaining 4 `[HIT_MAX_TOKENS]` books either.** Retried just those four
(the only ones where more budget could plausibly still be the
bottleneck -- the other zero-scoring books were already ruled out above
as language/content misses, not budget) at `max_tokens=12000`, same
`llama.cpp`/Metal/`n_ctx=40960` setup:

| Book | 6000 tok | 12000 tok |
| --- | --- | --- |
| `dnb-36942798X.pdf` | 288s, `[HIT_MAX_TOKENS]`, 0/0 | 677s, stopped on its own, 0/0 |
| `9783839458013.pdf` | 223s, `[HIT_MAX_TOKENS]`, 0/0 | 545s, `[HIT_MAX_TOKENS]`, 0/0 |
| `9781783742806.pdf` | 419s, `[HIT_MAX_TOKENS]`, 0/0 | 860s, `[HIT_MAX_TOKENS]`, 0/0 |
| `9781783743339.pdf` | 628s, `[HIT_MAX_TOKENS]`, 0/0 | 888s, `[HIT_MAX_TOKENS]`, 0/0 |

All four still score 0/0. Three still hit the (now doubled) cap, and the
fourth ran even longer (677s, up from 288s) before finally stopping on
its own -- still with no valid output. Doubling the budget roughly
doubled the wall-clock time these four burn without moving their score
at all, which is the signature of a genuine repetition/malformed-
generation failure, not a legitimate long-TOC book that just needs more
room. **Not worth raising the cap further** -- these four need a
different fix entirely (e.g. repetition-penalty sampling, detecting and
truncating a repeating n-gram mid-generation, or accepting them as a
known zero-recall cluster) rather than a bigger `max_tokens`. `f1=0.44`
at `max_tokens=6000` stands as the baseline; raising it beyond 6000 buys
nothing further on this corpus and should not be adopted as the
production/evaluation default.

### NuExtract-2.0-4B LoRA fine-tuning pilot: result (2026-08-14)

Ran the pilot for real on MPCDF Raven (see
`docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md`
for the design, `evaluation/hpc/README.md` for the HPC deployment itself
-- getting it running surfaced a long chain of environment/dependency
bugs, all fixed and documented there and in `nuextract.def`/
`run_pilot.slurm`'s own comments; nothing environment-specific belongs
here). Trained a LoRA adapter (rank 16, 4 epochs, gradient checkpointing)
on 78 books' TOC-scan-window text (89-book corpus, 78 train / 11 eval,
stratified split seed 42), merged, converted to GGUF Q4_K_M, and scored
both the fine-tuned and unmodified base checkpoint on the same 11-book
held-out split via the same `llama.cpp`-only scoring path
(`evaluate_nuextract_finetune.py`) -- an apples-to-apples comparison,
not directly comparable to the full-corpus `f1=0.44` baseline above
(different, much smaller subset).

| | precision | recall | f1 |
| --- | --- | --- | --- |
| Fine-tuned | 0.83 | 0.46 | **0.59** |
| Base (same split, same code) | 0.57 | 0.48 | **0.52** |

**Aggregate f1 improved (0.52 -> 0.59), but the per-book picture is more
complicated than "fine-tuning helps" -- it's propped up by big wins on a
few books while masking a real regression on two others:**

| Book | Fine-tuned f1 | Base f1 | Δ |
| --- | --- | --- | --- |
| copyrighted-scans/9783848736829 | 0.98 | 0.47 | +0.51 |
| copyrighted-scans/9783161538315 | 0.00 | 0.00 | — (both fail, pre-existing) |
| copyrighted-scans/9783428042241 | 0.95 | 0.94 | ~even |
| open-access/9781800641648 | **0.00** | 0.96 | **−0.96** |
| open-access/9781771993661 | 0.95 | 0.95 | ~even |
| open-access/9783839458013 | **0.00** | 0.30 | **−0.30** |
| open-access/9781906924874 | 0.96 | 0.90 | +0.06 |
| open-access/9783031466373 | 1.00 | 0.91 | +0.09 |
| open-access/9783907297285 | 0.96 | 0.96 | ~even |
| open-access/9781805111856 | 0.49 | 0.00 | +0.49 |
| open-access/9781805115717 | 0.00 | 0.00 | — (both fail, pre-existing) |

**Root cause of the two collapses: a decoding-time degenerate-repetition
loop, not a fine-tuning capability regression.** Added `--dump-dir` to
`evaluate_nuextract_finetune.py` (writes each book's raw completion
text, `finish_reason`, and parsed/expected chapters) to inspect why two
books scored 0/0 despite the model clearly having the right knowledge.
Both books' raw output showed `finish_reason: length` -- the model
correctly extracted several early chapters completely correctly (real
titles, real authors, real page numbers) before falling into an
infinite loop (`9781800641648`: repeating Hebrew transliteration
diacritics; `9783839458013`: repeating `…`) that burned the entire
`--max-tokens` budget without ever closing the JSON, so `parse_response`
saw truncated/invalid JSON and scored 0/0 -- not because the model didn't
know the answer, but because greedy decoding (`temperature=0.0`, no
repetition penalty) got stuck. `9783839458013` was already a known
repetition-prone book in the zero-shot baseline above (one of the four
`[HIT_MAX_TOKENS]` books that didn't recover even at `max_tokens=12000`)
-- this pilot didn't introduce that tendency, though `9781800641648` collapsing
is new (it scored 0.96 zero-shot in this same run).

**Tried fixing it with a repeat penalty; made the aggregate worse both
times.** The zero-shot baseline section above speculated
"repetition-penalty sampling" as a possible fix for exactly this failure
shape -- tested it for real here, twice:

| Config | `9781800641648` | `9783839458013` | 4 other previously-fine books | Aggregate f1 |
| --- | --- | --- | --- | --- |
| No penalty (baseline) | 0.00 | 0.00 | all 0.49-1.00 | **0.59** |
| `repeat_penalty=1.1`, 64-token window (llama-cpp-python default) | 0.83 | 0.09 | 3 collapsed to 0.00, 1 dropped to 0.62 | 0.41 |
| `repeat_penalty=1.1`, 16-token window | 0.38 | 0.22 | 2 still 0.00, 1 dropped to 0.56 | 0.34 |

Fixed the two target books (partially) but broke others every time: our
output is a JSON *list* of chapter dicts, repeating the same field names
(`"title"`/`"authors"`/`"printed_page_number"`) every ~20-40 tokens --
any blanket repeat penalty, at any window size tried, seems to disrupt
this model's ability to produce that legitimate, required repetition,
not just the genuine 1-4-token degenerate loops. Reverted to no penalty
(`--repeat-penalty`/`--repeat-last-n` remain available as documented,
off-by-default flags for future experimentation, not because they're
expected to work as-is). A more promising direction, if this failure
rate turns out to matter: detect the loop and salvage the valid JSON
prefix generated before it, at the application layer instead of the
sampler.

**Against the design spec's actual decision criterion -- a promising
but genuinely noisy signal, not a clean go.** The spec calls for
checking whether the *null-page-number rate* specifically dropped, not
just the aggregate f1; that specific check wasn't done here (would need
inspecting more of the `--dump-dir` output by hand across the 7 correctly
-scoring books), but the raw dumps for the two collapsed books are
suggestive -- every chapter extracted before the loop struck had a
correct, non-null `printed_page_number`, not the null-page-number
failure the pilot was meant to fix. Per the spec's own caution, an
11-book split is "powered to detect obviously helps' vs 'clearly
doesn't help,' not to measure a precise effect size" -- and this result
is neither: real wins on some books, a real new failure mode on two
others, aggregate f1 up but not overwhelmingly so. Worth a follow-up
decision (extend ground truth for a bigger/more stable split? pursue the
JSON-prefix-salvage fix for the repetition failures first?) rather than
either shipping this adapter or abandoning the approach on this result
alone.

## History

### Lessons learned from earlier models (2026-08-09)

Before settling on NuExtract-2.0-4B (below), two earlier/larger models in
the same family were evaluated and rejected. Kept as a short pointer so
they aren't re-tried blindly; full investigation detail for both has been
trimmed from this file since neither is in active use.

- **NuExtract-1.5-tiny: rejected, a genuine model-capability limit.**
  Zero-shot full 50-book run scored **0.00/0.00/0.00**. Root-caused (via
  direct Ollama API probing, ruling out a prompt/parsing bug in this
  repo, plus a follow-up retest through the original `transformers`
  checkpoint, ruling out GGUF-conversion fidelity as the cause) to the
  model having no instruction channel to ignore surrounding non-ToC
  noise: given a real book's scan window, it either stops generating
  immediately (empty output) or echoes the input back verbatim instead of
  filling the template. Even best-case, hand-curated ToC-only input (no
  realistic noise) only reached f1=0.09. Not a serving-path artifact --
  a real capability ceiling at the "tiny" size.
- **NuExtract3 (4B): worked well zero-shot, but dropped for deployment
  reasons, not accuracy.** A materially newer/larger model (Qwen3.5-4B-
  based) that closed nearly the entire gap: full 50-book run scored
  **f1=0.60** (open-access 0.69, copyrighted-scans 0.39) via `mlx-vlm`,
  competitive with or better than the cloud-LLM baseline's f1=0.43-0.48
  despite no fine-tuning and no `instructions` prompt-tuning. Dropped
  anyway after a CPU-only deployment check (the actual target: a no-GPU,
  16GB RAM, 4-vCPU Linux box, where MLX can't run at all and Ollama
  didn't yet support Qwen3.5-architecture GGUFs) found NuExtract-2.0-4B
  **1.42x faster** with equal-or-better accuracy (f1=0.97 vs 0.96 on a
  5-book `llama.cpp`/CPU sample) on that hardware profile, on top of
  already having mature Ollama support via its older, better-optimized
  Qwen2.5-VL architecture. All further work targets `numind/
  NuExtract-2.0-4B` only.

### Full 50-book, two-corpus run (GGUF Q4_K_M, Metal-offloaded)

Re-ran the full corpus through `llama.cpp` with `n_gpu_layers=-1` (Metal
offload for speed) and `n_ctx=40960` (raised from 8192 after one book's
noisy scan window exceeded it):

| Corpus | Books | Chapters | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| copyrighted-scans | 13 | 312 | 0.30 | 0.12 | 0.17 |
| open-access | 37 | 601 | 0.48 | 0.46 | 0.47 |
| **Total** | **50** | **913** | **0.45** | **0.35** | **0.39** |

Total wall time: 4156s (~69 min). This is NuExtract-2.0-4B's real
zero-shot ceiling on this corpus, through the same backend the target
deployment will use. It is noticeably below NuExtract3's own full-corpus
number on this same corpus (f1=0.60, see above) -- the smaller/older
model is less accurate zero-shot, which was already expected going in;
the CPU-comparison's f1=0.97 was a 5-book best-case sample, not
representative.

**Failure-mode breakdown.** Categorizing all 50 books by outcome:

| Failure type | Books | Share |
| --- | --- | --- |
| True truncation (empty/unparseable output) | 10 | 20% |
| Titles/authors correct, `printed_page_number` null on every entry | 14 | 28% |
| Low but nonzero | 7 | 14% |
| Good (f1 > 0.5) | 19 | 38% |

Truncation (10 books, several with 200-500s generation times before
producing no valid JSON -- the 1500-token output budget is too tight for
these books' larger TOCs) is real but is *not* the dominant cause of the
low aggregate. The larger group (14 books, 28%) is a different, more
specific failure: the model extracts titles and authors **verbatim
correctly** but leaves `printed_page_number` `null` for every entry, even
when the number is plainly present in the scan text next to the title.
Inspected two examples directly:

- `9783847432364.pdf` (German, clean extracted text): titles/authors
  match ground truth exactly; scan text clearly shows `"...das neue
  Gemeinsame   7"`, `"...gekämpft wird   15"` right next to each title,
  yet every predicted `printed_page_number` is `null`.
- `9780367439712.pdf` (English, but a badly OCR-scrambled multi-column
  contents page -- `"l"` for `"1"`, garbled column bleed): same pattern,
  titles correct, page numbers all `null`.

Since `match_toc_entries` requires an exact page-number match to count a
true positive, these 14 books score exactly 0 recall despite mostly-
correct extraction -- title-only accuracy is substantially better than
the f1=0.39 headline suggests. Two candidate sub-causes, not yet
disentangled: a likely non-English weakness (echoes the French-language
miss documented earlier for NuExtract3) and OCR-scrambled TOC layouts.
Unlike the truncation cluster or the earlier `transformers`/MPS bug, this
looks like a fixable extraction-formatting habit rather than a
fundamental capability gap -- the model already has the right
information, it just isn't attaching it to the record -- which is a
reasonable target for LoRA fine-tuning to move.

**This f1=0.39 is the baseline number a fine-tuning pilot needs to beat.**
