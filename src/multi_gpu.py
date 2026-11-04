"""Data-parallel SD-Turbo inference: one pipeline per GPU, CPU pool for vectorisation.

    GPU process 0 (cuda:0) --\\                          /-- CPU worker: vectorise + sanitise + validate
                              >-- result queue -- main --
    GPU process 1 (cuda:1) --/                          \\-- CPU worker: ...

Each GPU process loads its own pipeline and generates its shard of prompts. Images are sent
back as they finish and are vectorised by a CPU process pool while the GPUs keep generating.
torch is imported lazily so the pure helpers (split_prompts, resolve_devices) work without it.
"""
import argparse
import multiprocessing as mp
import queue
import statistics
import threading
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from kaggle_pkg.budget import DEFAULT_BUDGET_S, call_with_budget
from kaggle_pkg.model import Model
from kaggle_pkg.svg_sanitizer import MAX_BYTES, validate_svg

def resolve_devices(num_gpus="auto", available: int | None = None) -> list[str]:
    """Pick devices: 'auto' uses all visible GPUs, an int is capped at what's available,
    and with no GPUs (or num_gpus=0) we fall back to CPU."""
    if available is None:
        import torch
        available = torch.cuda.device_count()
    n = available if num_gpus in (None, "auto") else min(int(num_gpus), available)
    if num_gpus not in (None, "auto") and int(num_gpus) > available:
        warnings.warn(f"num_gpus={num_gpus} requested but only {available} available; using {max(n, 0)}")
    if n <= 0:
        return ["cpu"]
    return [f"cuda:{i}" for i in range(n)]

def split_prompts(prompts: list[str], n: int) -> list[list[tuple[int, str]]]:
    """Round-robin split of (index, prompt) pairs into n shards, so shard sizes differ by at most 1."""
    return [[(i, p) for i, p in enumerate(prompts) if i % n == k] for k in range(n)]

def _percentile(xs, q):
    if not xs: return float("nan")
    return float(np.percentile(xs, q))

def _gpu_worker(device, items, dcfg, budget_s, warmup, out_q, start_evt):
    """Runs in its own (spawned) process: load pipeline on `device`, generate its shard."""
    try:
        import torch
        from src.diffusion_pipeline import load_pipeline
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        pipe, info = load_pipeline(dcfg["model_id"], device, dcfg.get("dtype", "auto"),
                                   dcfg.get("use_xformers", True), dcfg.get("attention_slicing", True))
        kw = dict(num_inference_steps=int(dcfg["steps"]), guidance_scale=float(dcfg["guidance"]),
                  height=int(dcfg["height"]), width=int(dcfg["width"]))
        if warmup:  # first call pays CUDA/kernel init costs; keep it out of the timings
            pipe("warmup", **kw)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        out_q.put(("ready", device, info))
        start_evt.wait()

        lock = threading.Lock()  # a timed-out call keeps running; don't let the next one overlap it
        def generate(prompts, gens):
            with lock:
                return pipe(list(prompts), generator=gens, **kw).images

        bs, seed = max(1, int(dcfg.get("batch_size", 1))), int(dcfg.get("seed", 42))
        for b in range(0, len(items), bs):
            batch = items[b:b + bs]
            idxs, prompts = zip(*batch)
            gens = [torch.Generator(device=device).manual_seed(seed + i) for i in idxs]
            t = time.perf_counter()
            images, timed_out = call_with_budget(lambda: generate(prompts, gens), budget_s * len(batch),
                                                 fallback=lambda: [None] * len(batch), label=f"{device} batch {idxs}")
            per_prompt_s = (time.perf_counter() - t) / len(batch)
            for i, p, img in zip(idxs, prompts, images):
                arr = None if img is None else np.asarray(img.convert("RGB"))
                out_q.put(("image", i, p, arr, per_prompt_s, timed_out, device))

        stats = dict(info)
        if device.startswith("cuda"):
            stats["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            stats["gpu_name"] = torch.cuda.get_device_name(device)
        out_q.put(("done", device, stats))
    except Exception:
        out_q.put(("error", device, traceback.format_exc()))

def postprocess(idx, prompt, img_rgb, gen_s, timed_out, device, vcfg, svg_dir=None, images_dir=None):
    """CPU side: vectorise + sanitise + validate one image; fall back to the heuristic
    renderer on timeout or if the SVG fails validation. Returns a per-prompt record."""
    import cv2
    from src.vectorize import vectorize_array
    t = time.perf_counter()
    W, H = vcfg["canvas_size"]
    raw_bytes, reason, svg = None, None, None
    if img_rgb is None:
        reason = "timeout"
    else:
        if images_dir:
            from PIL import Image
            Image.fromarray(img_rgb).save(Path(images_dir) / f"{idx:05d}.png")
        svg, raw_bytes = vectorize_array(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), vcfg)
        problems = validate_svg(svg)
        if problems:
            reason = "invalid: " + "; ".join(problems[:3])
    diffusion_valid = reason is None
    if reason:
        warnings.warn(f"prompt {idx} ({reason}); using heuristic renderer")
        svg = Model({"width": W, "height": H})._heuristic_svg(prompt)
    if svg_dir:
        (Path(svg_dir) / f"{idx:05d}.svg").write_text(svg, encoding="utf-8")
    return {"idx": idx, "prompt": prompt, "device": device, "gen_s": gen_s,
            "vec_s": time.perf_counter() - t, "raw_bytes": raw_bytes, "svg_bytes": len(svg.encode("utf-8")),
            "diffusion_valid": diffusion_valid, "final_valid": not validate_svg(svg), "fallback": reason}

def _next_msg(q, procs):
    while True:
        try:
            return q.get(timeout=5)
        except queue.Empty:
            dead = [p for p in procs if p.exitcode not in (None, 0)]
            if dead:
                raise RuntimeError(f"GPU worker exited with code {dead[0].exitcode}")

def run(prompts: list[str], cfg: dict, svg_dir=None, images_dir=None, warmup=True) -> dict:
    """Generate + vectorise all prompts. Returns per-prompt records, per-device stats and the
    timed window (t_start / t_gen_end as time.time()) for external GPU monitoring."""
    dcfg, vcfg = cfg["diffusion"], cfg["vectorize"]
    rcfg = cfg.get("runtime", {})
    budget_s = float(rcfg.get("time_budget_s", DEFAULT_BUDGET_S))
    for d in (svg_dir, images_dir):
        if d: Path(d).mkdir(parents=True, exist_ok=True)

    devices = resolve_devices(dcfg.get("num_gpus", "auto"))
    shards = [(d, s) for d, s in zip(devices, split_prompts(prompts, len(devices))) if s]
    ctx = mp.get_context("spawn")  # CUDA cannot be re-initialised in forked children
    out_q, start_evt = ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=_gpu_worker, args=(d, s, dcfg, budget_s, warmup, out_q, start_evt), daemon=True)
             for d, s in shards]
    for p in procs: p.start()

    device_stats, futures = {}, []
    try:
        for _ in procs:  # wait until every pipeline is loaded and warmed up
            kind, device, payload = _next_msg(out_q, procs)
            if kind == "error": raise RuntimeError(f"{device} failed:\n{payload}")
            device_stats[device] = payload
        with ProcessPoolExecutor(int(rcfg.get("cpu_workers", 2)), mp_context=ctx) as pool:
            t_start = time.time()
            start_evt.set()
            done = 0
            while done < len(procs):
                msg = _next_msg(out_q, procs)
                if msg[0] == "image":
                    futures.append(pool.submit(postprocess, *msg[1:], vcfg, svg_dir, images_dir))
                elif msg[0] == "done":
                    done += 1; device_stats[msg[1]] = msg[2]
                else:
                    raise RuntimeError(f"{msg[1]} failed:\n{msg[2]}")
            t_gen_end = time.time()
            records = sorted((f.result() for f in futures), key=lambda r: r["idx"])
        t_end = time.time()
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive(): p.terminate()

    return {"devices": [d for d, _ in shards], "records": records, "device_stats": device_stats,
            "t_start": t_start, "t_gen_end": t_gen_end, "t_end": t_end}

def summarize(result: dict) -> dict:
    """Throughput / latency / constraint metrics from run() output."""
    recs = result["records"]
    n = len(recs)
    latency = [r["gen_s"] + r["vec_s"] for r in recs]
    gen_window = result["t_gen_end"] - result["t_start"]
    return {
        "n_prompts": n,
        "images_per_s": n / gen_window if gen_window > 0 else float("nan"),
        "e2e_s": result["t_end"] - result["t_start"],
        "sec_per_prompt_p50": _percentile(latency, 50),
        "sec_per_prompt_p95": _percentile(latency, 95),
        "max_sec_per_prompt": max(latency) if latency else float("nan"),
        # share of diffusion-derived SVGs that pass validation (<=10KB, allowlist) without fallback
        "svg_pass_rate": sum(r["diffusion_valid"] for r in recs) / n if n else float("nan"),
        # share that were already <=10KB before the sanitizer dropped paths
        "raw_under_10kb_rate": statistics.mean(r["raw_bytes"] is not None and r["raw_bytes"] <= MAX_BYTES for r in recs) if n else float("nan"),
        "final_valid_rate": sum(r["final_valid"] for r in recs) / n if n else float("nan"),
        "fallback_rate": sum(r["fallback"] is not None for r in recs) / n if n else float("nan"),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-GPU text -> PNG -> SVG")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    prompts = [p.strip() for p in Path(cfg["data"]["prompts_file"]).read_text(encoding="utf-8").splitlines() if p.strip()]
    res = run(prompts, cfg, svg_dir=cfg["paths"]["svg_dir"], images_dir=cfg["paths"]["images_dir"])
    print(f"devices={res['devices']}", summarize(res))
