# Drawing with LLMs — SVG Generator (Package + Local Pipeline)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USER>/<YOUR_REPO>/blob/main/notebooks/colab_pipeline.ipynb)

This repo turns text prompts into **valid SVG** for the *Drawing with LLMs* competition. It includes:

- A **Kaggle Package** (`kaggle_pkg`) exposing a `Model.predict(prompt) -> svg_str`.
- A **local/Colab pipeline** to generate rasters via diffusion and vectorize to strict SVG.

## Why two paths?

- **Kaggle Package**: must run with **Internet disabled**, under **time limits** (≤5 min/prompt, ≤9h total), and pass **SVG constraints** (≤10KB, allowlisted tags/attrs, no external data). The package here uses a **fast, deterministic heuristics-based renderer** (shapes from keywords) and a **sanitizer** to stay valid.
- **Local Pipeline**: your Kaggle notebook likely used a diffusion model (e.g., SD Turbo) → **PNG** → vectorization. That path is included in `src/` for Colab or local experiments. It’s too heavy to run per-prompt in the Package environment, but great for R&D and generating training assets.

---

## Repo Layout
```text
drawing-with-llms-svg/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ default.yaml
├─ prompts/
│  └─ prompts.txt
├─ kaggle_pkg/
│  ├─ __init__.py
│  ├─ model.py          # Model.predict(prompt) -> SVG (fast, deterministic)
│  └─ svg_sanitizer.py  # strict allowlist & 10KB cap
├─ src/
│  ├─ diffusion_pipeline.py  # text→PNG with diffusers
│  ├─ vectorize.py           # PNG→SVG (path-only)
│  └─ pipeline.py            # both steps in one
└─ notebooks/
   └─ colab_pipeline.ipynb
```

## Quickstart (Local / Colab)
```bash
pip install -r requirements.txt
# Generate rasters
python -m src.diffusion_pipeline --config configs/default.yaml
# Vectorize to SVG
python -m src.vectorize --config configs/default.yaml
# Or run both:
python -m src.pipeline --config configs/default.yaml
```

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

> Note: If you want to experiment locally with your diffusion→vector pipeline inside `Model`, set env `DIFFUSION_TO_SVG=1`. It will try a quick SD Turbo + vectorize path. **Do not enable this for package inference on Kaggle** (too heavy).

### Packaging Notes (Kaggle)
- Internet is **disabled**. Attach any model weights as Kaggle Datasets in your Notebook and reference them via `/kaggle/input/...` paths (not included here).
- Keep `predict()` time **< 5 minutes per prompt**.
- Keep SVG **< 10KB** and within **allowed elements/attributes**. This repo’s sanitizer is conservative (path-only).

## Competition Considerations

**Constraints enforcement**  
- Our sanitizer emits only `<svg>` and `<path>` (plus optional `<g>`), with a small attribute allowlist (`d, fill, stroke, stroke-width, fill-rule, opacity`).  
- No `<style>`, `<image>`, external links, or `foreignObject`.  
- Max size enforced at **10,000 bytes**; excess paths are dropped.

**Evaluation pipeline (what matters)**  
- SVG → PNG via **cairosvg** (in official eval). Then VQA (PaliGemma) for faithfulness (**TIFA**), OCR penalty for text, and **CLIP Aesthetic** score. Final is the **harmonic mean** favoring VQA.  
- Implications: prioritize **composition** and **object correctness** over ornate textures; avoid unintended **text-like artifacts**; keep images clean.

**Performance**  
- For raster→vector experiments, prefer **SD Turbo** (1–4 steps). On Colab **A100**, defaults work. On **T4 16GB**, reduce resolution/steps.

## Configuration – `configs/default.yaml`

See comments in the file for each field. Tweak diffusion model/steps and vectorization thresholds to trade detail vs. simplicity.

## Colab
Use the badge above (update `<YOUR_USER>/<YOUR_REPO>` after pushing). The notebook runs the local pipeline with the YAML config.


---
**Next steps I can do for you**  
- Add a strict allowlist closer to the official `svg_constraints` package.  
- Add a tiny rule-based library of icons to improve VQA scores.  
- Set up a minimal **Kaggle Package notebook** template that builds and submits automatically.
