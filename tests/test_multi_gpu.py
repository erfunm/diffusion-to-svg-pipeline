import time
import warnings

import pytest

from kaggle_pkg.budget import call_with_budget
from kaggle_pkg.svg_sanitizer import validate_svg
from src.multi_gpu import postprocess, resolve_devices, split_prompts

VCFG = dict(canny_low=80, canny_high=160, min_area=150, approx_epsilon=2.0, max_paths=128,
            canvas_size=[512, 512], use_fill=True, use_stroke=False)

@pytest.mark.parametrize("n_prompts,n_shards", [(10, 2), (7, 2), (1, 2), (5, 1), (3, 4)])
def test_split_prompts_covers_each_prompt_once_and_balances(n_prompts, n_shards):
    prompts = [f"p{i}" for i in range(n_prompts)]
    shards = split_prompts(prompts, n_shards)
    assert len(shards) == n_shards
    flat = sorted(i for s in shards for i, _ in s)
    assert flat == list(range(n_prompts))
    assert all(prompts[i] == p for s in shards for i, p in s)
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1

def test_resolve_devices():
    assert resolve_devices("auto", available=2) == ["cuda:0", "cuda:1"]
    assert resolve_devices(2, available=2) == ["cuda:0", "cuda:1"]
    assert resolve_devices(1, available=2) == ["cuda:0"]
    assert resolve_devices(0, available=2) == ["cpu"]
    with pytest.warns(UserWarning):
        assert resolve_devices(2, available=1) == ["cuda:0"]

def test_cpu_fallback_when_no_gpu():
    assert resolve_devices("auto", available=0) == ["cpu"]
    with pytest.warns(UserWarning):
        assert resolve_devices(2, available=0) == ["cpu"]

def test_cpu_fallback_with_real_torch():
    torch = pytest.importorskip("torch")
    from src.diffusion_pipeline import resolve_dtype
    if not torch.cuda.is_available():
        assert resolve_devices("auto") == ["cpu"]
    with pytest.warns(UserWarning):
        assert resolve_dtype("fp16", "cpu") == torch.float32
    assert resolve_dtype("auto", "cpu") == torch.float32
    assert resolve_dtype("auto", "cuda:1") == torch.float16

def test_postprocess_timeout_falls_back_to_heuristic(tmp_path):
    with pytest.warns(UserWarning):
        rec = postprocess(3, "a red circle", None, 301.0, True, "cuda:0", VCFG, svg_dir=tmp_path)
    assert rec["fallback"] == "timeout" and not rec["diffusion_valid"] and rec["final_valid"]
    assert validate_svg((tmp_path / "00003.svg").read_text()) == []

def test_call_with_budget():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert call_with_budget(lambda: 1, 5, fallback=lambda: 0) == (1, False)
    with pytest.warns(UserWarning, match="budget"):
        assert call_with_budget(lambda: time.sleep(1) or 1, 0.05, fallback=lambda: 0) == (0, True)
