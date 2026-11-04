# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Text prompt → valid SVG for the Kaggle *Drawing with LLMs* competition. Two code paths:

1. **`kaggle_pkg/`**: the submission package. Kaggle imports it and calls `Model().predict(prompt) -> str`. It runs with no internet access, a limit of ≤5 min per prompt and ≤9 h in total, and its output must be ≤10KB of allowlisted SVG. It must stay importable without torch, diffusers or cv2; heavy imports are lazy and live inside `_diffusion_to_svg`.
2. **`src/`**: the SD-Turbo → OpenCV → SVG pipeline, data-parallel across GPUs, plus a benchmark. `src/` imports from `kaggle_pkg`; never the reverse.

## Commands

Python 3.10+. Install `torch` separately to match your CUDA version. Run everything from the repo root, because config paths are relative to the current working directory.

```bash
pip install -r requirements.txt
python -m kaggle_pkg.model --prompt "a red triangle icon"        # heuristic submission path
python -m src.pipeline --config configs/default.yaml              # multi-GPU end to end (CPU fallback)
python -m src.benchmark --config configs/default.yaml [--only 2gpu_fp16 ...] [--n-prompts 8]
python -m src.check_svgs outputs/svg                              # non-zero exit on violations

# Lint + tests (CPU only, same as .github/workflows/ci.yml)
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-dev.txt
ruff check . && pytest -q
pytest tests/test_multi_gpu.py::test_split_prompts_covers_each_prompt_once_and_balances   # single test
```

Ruff rules are pinned in `pyproject.toml` (E4/E7/E9/F, with the one-liner style E401/E701/E702 allowed). Tests do not need diffusers or a GPU.

## Architecture notes

- **Multi-GPU (`src/multi_gpu.py`)**: one `spawn`ed process per device, each with its own pipeline. Prompts are split round-robin, and prompt *i* always uses seed `seed + i`. Workers send `("ready" | "image" | "done" | "error", ...)` messages on a single queue. The main process waits for every worker to be ready (pipeline loaded and warmed up), then sets a start event, then sends each image to a CPU `ProcessPoolExecutor` running `postprocess` (vectorise → sanitise → validate → heuristic fallback). `run()` returns timestamps (`t_start`, `t_gen_end`) that `benchmark.py` uses to filter NVML and nvidia-smi samples. Anything that runs in a child process must be a module-level function, and any script that calls `run()` needs an `if __name__ == "__main__"` guard.
- **Attention settings (`src/diffusion_pipeline.load_pipeline`)**: xFormers and attention slicing each *replace* the diffusers attention processor. Slicing is applied first and xFormers second, so xFormers wins when it is available. The function returns `info` describing what is actually active (`attn_processor`, `xformers_active`, `dtype`). Report that, not the requested flags. Half precision on CPU falls back to fp32. Don't wrap calls in `torch.autocast`, because it would make "fp32" runs mixed precision.
- **Constraints (`kaggle_pkg/svg_sanitizer.py`)**: `sanitize_svg` works with regexes and rebuilds output from `<path>` elements only. It keeps paths in order until it reaches `MAX_BYTES`, so emit the most important paths first (`vectorize_array` sorts by area). `validate_svg` is an independent check that parses the XML; every pipeline output goes through it.
- **Time budget (`kaggle_pkg/budget.py`)**: `call_with_budget` runs a function in a thread and returns `(fallback(), True)` on timeout. The timed-out thread can't be killed, so anything that shares a pipeline with it must hold a lock (both `Model` and `_gpu_worker` do this).
- **Two vectorisers**: `src/vectorize.vectorize_array` (mean-colour fills, configurable) and `Model._diffusion_to_svg` (black fills, hardcoded parameters) are separate copies; the package can't import `src/`.

## Benchmark honesty

`reports/benchmark.csv` is the only allowed source for performance numbers in the README. The README's Performance table holds `[TO BE FILLED FROM KAGGLE RUN]` placeholders until a real Kaggle T4 x2 run has produced the CSV. Never invent or estimate values.
