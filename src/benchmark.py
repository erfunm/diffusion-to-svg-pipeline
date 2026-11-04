"""Benchmark SD-Turbo -> SVG across GPU count, precision and attention settings.

Every configuration uses the same prompts and per-prompt seeds. Writes:
  reports/benchmark.csv          one row per configuration (skipped rows say why)
  reports/figures/benchmark.png  throughput and peak memory per configuration
  reports/nvidia_smi_dual.txt    nvidia-smi snapshots taken during the dual-GPU run

    python -m src.benchmark --config configs/default.yaml
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # make CUDA indices match NVML / nvidia-smi

import argparse
import copy
import csv
import subprocess
import threading
import time
from pathlib import Path

import yaml

from .multi_gpu import run, summarize

# (name, diffusion overrides). Single vs dual, fp32 vs fp16, xFormers on/off, slicing on/off.
CONFIGS = [
    ("1gpu_fp32",                  dict(num_gpus=1, dtype="fp32", use_xformers=False, attention_slicing=False)),
    ("1gpu_fp16",                  dict(num_gpus=1, dtype="fp16", use_xformers=False, attention_slicing=False)),
    ("1gpu_fp16_slicing",          dict(num_gpus=1, dtype="fp16", use_xformers=False, attention_slicing=True)),
    ("1gpu_fp16_xformers",         dict(num_gpus=1, dtype="fp16", use_xformers=True,  attention_slicing=False)),
    ("1gpu_fp16_xformers_slicing", dict(num_gpus=1, dtype="fp16", use_xformers=True,  attention_slicing=True)),
    ("2gpu_fp16",                  dict(num_gpus=2, dtype="fp16", use_xformers=False, attention_slicing=False)),
    ("2gpu_fp16_xformers",         dict(num_gpus=2, dtype="fp16", use_xformers=True,  attention_slicing=False)),
]

class GpuMonitor:
    """Background sampler: NVML utilisation per GPU, plus optional nvidia-smi snapshots.
    Samples are timestamped so we can keep only the ones inside the timed window."""

    def __init__(self, interval_s=0.2, smi_interval_s=None):
        self.interval_s, self.smi_interval_s = interval_s, smi_interval_s
        self.util, self.smi = [], []  # (t, [util%...]), (t, text)
        self._stop = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as e:
            print(f"[benchmark] NVML unavailable, GPU utilisation will be blank: {e}")
            self._nvml, self._handles = None, []

    def _loop(self):
        last_smi = 0.0
        while not self._stop.is_set():
            now = time.time()
            if self._handles:
                self.util.append((now, [self._nvml.nvmlDeviceGetUtilizationRates(h).gpu for h in self._handles]))
            if self.smi_interval_s and now - last_smi >= self.smi_interval_s:
                last_smi = now
                try:
                    self.smi.append((now, subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout))
                except Exception as e:
                    self.smi.append((now, f"nvidia-smi failed: {e}"))
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def mean_util(self, t0, t1) -> list[float]:
        rows = [u for t, u in self.util if t0 <= t <= t1]
        if not rows: return []
        return [sum(col) / len(col) for col in zip(*rows)]

def load_prompts(path, n=None):
    prompts = [p.strip() for p in Path(path).read_text(encoding="utf-8").splitlines() if p.strip()]
    return prompts[:n] if n else prompts

def benchmark_config(name, overrides, base_cfg, prompts, available_gpus, out_root, smi_path=None):
    import torch
    cfg = copy.deepcopy(base_cfg)
    cfg["diffusion"].update(overrides)
    row = {"config": name, **{k: overrides[k] for k in ("num_gpus", "dtype", "use_xformers", "attention_slicing")},
           "batch_size": cfg["diffusion"].get("batch_size", 1), "torch": torch.__version__}
    if overrides["num_gpus"] > available_gpus:
        row["status"] = f"skipped: needs {overrides['num_gpus']} GPU(s), {available_gpus} available"
        return row

    with GpuMonitor(smi_interval_s=1.0 if smi_path else None) as mon:
        res = run(prompts, cfg, svg_dir=out_root / name)
    t0, t1 = res["t_start"], res["t_gen_end"]
    row.update(status="ok", devices=" ".join(res["devices"]), **summarize(res))
    util = mon.mean_util(t0, t1)
    for i in range(2):
        st = res["device_stats"].get(f"cuda:{i}", {})
        row[f"gpu{i}_name"] = st.get("gpu_name", "")
        row[f"peak_mem_mb_gpu{i}"] = round(st["peak_mem_mb"], 1) if "peak_mem_mb" in st else ""
        row[f"util_pct_gpu{i}"] = round(util[i], 1) if i < len(util) else ""
    first = next(iter(res["device_stats"].values()))
    row.update(dtype_active=first["dtype"], xformers_active=first["xformers_active"], attn_processor=first["attn_processor"])

    if smi_path:
        snaps = [(t, s) for t, s in mon.smi if t0 <= t <= t1] or mon.smi  # keep all if run was shorter than 1s
        with open(smi_path, "w", encoding="utf-8") as f:
            f.write(f"# nvidia-smi during '{name}' (devices: {row['devices']})\n")
            f.write(f"# timed window: {time.strftime('%H:%M:%S', time.localtime(t0))} - "
                    f"{time.strftime('%H:%M:%S', time.localtime(t1))}, {len(snaps)} snapshot(s)\n\n")
            for t, s in snaps:
                f.write(f"=== {time.strftime('%H:%M:%S', time.localtime(t))} "
                        f"({'inside' if t0 <= t <= t1 else 'outside'} timed window) ===\n{s}\n")
    return row

def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok: return
    names = [r["config"] for r in ok]
    mem = [max([r[f"peak_mem_mb_gpu{i}"] for i in range(2) if r.get(f"peak_mem_mb_gpu{i}") != ""] or [0]) / 1024 for r in ok]
    ink, muted, grid, bar = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(ok) + 1.6), sharey=True, facecolor="#fcfcfb")
    panels = [(axes[0], [r["images_per_s"] for r in ok], "Throughput (images/s, higher is better)", "{:.2f}"),
              (axes[1], mem, "Peak GPU memory per device (GiB)", "{:.1f}")]
    for ax, vals, title, fmt in panels:
        ax.set_facecolor("#fcfcfb")
        ax.barh(names, vals, color=bar, height=0.55)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.grid(axis="x", color=grid, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, length=0)
        for y, v in enumerate(vals):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9, color=muted)
        ax.margins(x=0.15)
    axes[0].invert_yaxis()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())

def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompts", default="prompts/benchmark_prompts.txt")
    ap.add_argument("--n-prompts", type=int, default=None, help="use the first N prompts")
    ap.add_argument("--only", nargs="*", help="subset of config names to run")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    prompts = load_prompts(args.prompts, args.n_prompts)
    reports = Path(args.reports)
    available = torch.cuda.device_count()
    print(f"[benchmark] {len(prompts)} prompts, {available} GPU(s): "
          f"{[torch.cuda.get_device_name(i) for i in range(available)]}")

    rows = []
    for name, overrides in CONFIGS:
        if args.only and name not in args.only: continue
        smi = reports / "nvidia_smi_dual.txt" if overrides["num_gpus"] == 2 else None  # last dual run wins
        print(f"[benchmark] {name} ...", flush=True)
        try:
            row = benchmark_config(name, overrides, base, prompts, available, Path("outputs/benchmark"), smi)
        except Exception as e:  # e.g. OOM in fp32: record it rather than abort the whole sweep
            row = {"config": name, **overrides, "status": f"error: {type(e).__name__}: {str(e).splitlines()[0][:200]}"}
        print(f"[benchmark] {name}: {row}", flush=True)
        rows.append(row)

    reports.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(reports / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    plot(rows, reports / "figures" / "benchmark.png")
    print(f"[benchmark] wrote {reports/'benchmark.csv'} and {reports/'figures'/'benchmark.png'}")

if __name__ == "__main__":
    main()
