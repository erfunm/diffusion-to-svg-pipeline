# Drawing with LLMs — SVG Generator (Package + Local Pipeline)

This repo turns text prompts into **valid SVG** for the *Drawing with LLMs* competition. It includes:

- A **Kaggle Package** (`kaggle_pkg`) exposing a `Model.predict(prompt) -> svg_str`.
- A **diffusion → SVG pipeline** (`src/`): SD-Turbo generates rasters (data-parallel across GPUs), then OpenCV vectorises them into strict, sanitised SVG.

## Why two paths?

- **Kaggle Package**: must run with **Internet disabled**, under **time limits** (≤5 min/prompt, ≤9h total), and pass **SVG constraints** (≤10KB, allowlisted tags/attrs, no external data). By default the package uses a **fast, deterministic heuristics-based renderer** (shapes from keywords) and a **sanitizer** to stay valid.
- **Diffusion pipeline**: SD-Turbo → **PNG** → vectorisation. Use it on Kaggle/Colab GPUs for R&D, benchmarking and generating assets. It downloads weights from the Hugging Face Hub unless `model_id` points to a local path (e.g. an attached Kaggle dataset).

---

## Repo Layout
```text
diffusion-to-svg-pipeline/
├─ configs/default.yaml        # diffusion, multi-GPU, runtime budget, vectorisation
├─ prompts/
│  ├─ prompts.txt              # pipeline input
│  └─ benchmark_prompts.txt    # fixed prompt set for the benchmark
├─ kaggle_pkg/
│  ├─ model.py                 # Model.predict(prompt) -> SVG (fast, deterministic)
│  ├─ svg_sanitizer.py         # sanitize_svg (allowlist + 10KB cap), validate_svg (independent check)
│  └─ budget.py                # per-prompt time budget with fallback
├─ src/
│  ├─ diffusion_pipeline.py    # load_pipeline (FP16 / xFormers / attention slicing), single-device generation
│  ├─ multi_gpu.py             # one pipeline per GPU + CPU vectorisation pool
│  ├─ vectorize.py             # PNG -> SVG (Canny, contours, approxPolyDP), sanitised
│  ├─ pipeline.py              # end-to-end run
│  ├─ benchmark.py             # GPU/precision/attention benchmark -> reports/
│  └─ check_svgs.py            # validate a directory of SVGs
├─ notebooks/
│  ├─ colab_pipeline.ipynb
│  └─ kaggle_dual_t4.ipynb     # one-click benchmark on Kaggle "GPU T4 x2"
├─ reports/                    # benchmark outputs (created by src/benchmark.py)
└─ tests/                      # pytest, CPU-only
```

## Quickstart (Local / Colab / Kaggle)
```bash
pip install -r requirements.txt          # install torch separately to match your CUDA
# Full pipeline on all visible GPUs (falls back to CPU if none)
python -m src.pipeline --config configs/default.yaml
# Individual stages (single device)
python -m src.diffusion_pipeline --config configs/default.yaml
python -m src.vectorize --config configs/default.yaml
# Check every output SVG against the constraints (non-zero exit on failure)
python -m src.check_svgs outputs/svg
# Benchmark (writes reports/)
python -m src.benchmark --config configs/default.yaml
```

Tests and lint (CPU only, as in CI):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

## How dual-GPU works

`src/multi_gpu.py` uses data parallelism rather than model parallelism. SD-Turbo fits comfortably on one 16 GB T4, so each GPU gets a full copy of the pipeline and half of the prompts.

1. `resolve_devices` picks `cuda:0..N-1` (`num_gpus: auto` means all visible GPUs; requests are capped at what's available). With no GPU it falls back to `cpu`.
2. `split_prompts` splits the prompts round-robin across devices. Prompt *i* always uses seed `seed + i`, so the images don't depend on which GPU generated them.
3. One process per GPU is started with the `spawn` start method. Each process loads its own pipeline with the configured dtype, attention slicing and xFormers, runs a warm-up call, then waits on a shared start event so all GPUs begin the timed run together.
4. Each finished image goes into a queue. The main process hands it to a **CPU process pool**, which vectorises, sanitises and validates it, so the GPUs never wait on OpenCV.

xFormers and attention slicing are **not stacked**: in diffusers each replaces the attention processor. Slicing is applied first and xFormers second, so xFormers is used when it imports and slicing is the fallback when it doesn't. `load_pipeline` reports the processor actually in use (`attn_processor`, `xformers_active`), and the benchmark records it.

## Constraints

- **≤10 KB, allowlisted SVG.** `sanitize_svg` keeps only `<path>` elements under a fresh `<svg>` root, with attributes `d, fill, stroke, stroke-width, fill-rule, opacity`. It removes `<style>`, `<script>`, `<image>`, `foreignObject`, comments and external references. If the output is over 10,000 bytes it keeps paths in order until the cap is reached; the vectoriser emits the largest-area paths first, so small details are dropped first.
- **Independent check.** `validate_svg` parses the final SVG with an XML parser and re-checks size, tags and attributes. The pipeline runs it on every output and falls back to the heuristic renderer if it fails. `python -m src.check_svgs <dir>` runs the same check on a directory.
- **≤5 min per prompt.** Generation is wrapped by `call_with_budget` (`runtime.time_budget_s`, default 300 s). On timeout it warns and returns the heuristic SVG for that prompt. In `Model.predict` this applies to the optional diffusion path (`time_budget_s` in the config or the `SVG_TIME_BUDGET_S` env var). A timed-out call can't be killed, so it finishes in the background behind a lock while the next prompt waits.
- **Offline.** The package's default heuristic path has no network or model dependencies. The diffusion path is offline only when weights are loaded from a local or attached path.

**Evaluation pipeline (what matters).** SVG is rendered to PNG with **cairosvg**, then scored with VQA (PaliGemma, **TIFA**) for faithfulness, an OCR penalty for text, and a **CLIP aesthetic** score. The final score is a **harmonic mean** that weights VQA most heavily. In practice this means: prioritise **composition** and **object correctness** over ornate textures, and avoid unintended **text-like artifacts**.

## Performance

All numbers in this section must come from `reports/benchmark.csv`, produced by `python -m src.benchmark` on Kaggle **GPU T4 x2** (`notebooks/kaggle_dual_t4.ipynb`). The setup is 32 prompts from `prompts/benchmark_prompts.txt`, 512×512, 4 steps, batch size 1, with a warm-up excluded from timings.

| Config | GPUs | dtype | Attention processor | Images/s | s/prompt p50 | s/prompt p95 | Peak mem GPU0 (MB) | Peak mem GPU1 (MB) | Util GPU0 / GPU1 (%) | SVG pass rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 1gpu_fp32 | 1 | fp32 | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | – | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] |
| 1gpu_fp16 | 1 | fp16 | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | – | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] |
| 1gpu_fp16_slicing | 1 | fp16 | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | – | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] |
| 1gpu_fp16_xformers | 1 | fp16 | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | – | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] |
| 1gpu_fp16_xformers_slicing | 1 | fp16 | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | – | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] |
| 2gpu_fp16 | 2 | fp16 | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] |
| 2gpu_fp16_xformers | 2 | fp16 | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] | [TO BE FILLED FROM KAGGLE RUN] |

Column definitions (all from `reports/benchmark.csv`):
- **Images/s**: prompts ÷ wall time from the synchronised start until the last image is generated.
- **s/prompt**: generation time (per batch ÷ batch size) plus CPU vectorisation time.
- **Peak mem**: `torch.cuda.max_memory_allocated` per device, reset after warm-up.
- **Util**: mean NVML utilisation during the timed window.
- **SVG pass rate**: the share of diffusion-derived SVGs that pass `validate_svg` (≤10 KB, allowlist) without needing the fallback. `raw_under_10kb_rate` in the CSV shows how often the size cap was reached before sanitisation.

Evidence that both GPUs were used: `reports/nvidia_smi_dual.txt`. Chart: `reports/figures/benchmark.png`.

## Kaggle Package (Submission) – `kaggle_pkg`

Kaggle will import your package and call:

```python
from kaggle_pkg import Model
m = Model()
svg = m.predict("a red triangle icon")
```

This implementation keeps things **valid, small, and fast**:

- **Shape heuristics**: simple geometric icons from keywords (circle, square, triangle, star, color words).
- **Sanitization**: `kaggle_pkg/svg_sanitizer.py` removes disallowed tags/attrs and enforces **≤10KB** output.
- **No external refs, no raster data, no styles**.

> Note: to experiment locally with the diffusion → vector path inside `Model`, set the env var `DIFFUSION_TO_SVG=1`. It loads SD Turbo once and is bounded by the time budget. **Do not enable this for package inference on Kaggle** (too heavy, and it needs model weights).

### Packaging Notes (Kaggle)
- Internet is **disabled**. Attach any model weights as Kaggle Datasets in your Notebook and reference them via `/kaggle/input/...` paths (not included here).
- Keep `predict()` time **< 5 minutes per prompt**.
- Keep SVG **< 10KB** and within **allowed elements/attributes**. This repo's sanitizer is conservative (path-only).

## Configuration – `configs/default.yaml`

See comments in the file for each field. `diffusion.*` controls the model, precision (`dtype`), `num_gpus`, `batch_size`, `use_xformers` and `attention_slicing`. `runtime.*` sets the per-prompt time budget and the number of CPU vectorisation workers. `vectorize.*` trades detail against simplicity.
