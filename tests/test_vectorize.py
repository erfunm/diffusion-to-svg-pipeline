import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from kaggle_pkg.svg_sanitizer import MAX_BYTES, validate_svg  # noqa: E402
from src.vectorize import vectorize_array  # noqa: E402

VCFG = dict(canny_low=80, canny_high=160, min_area=150, approx_epsilon=2.0, max_paths=128,
            canvas_size=[512, 512], use_fill=True, use_stroke=False)

def test_vectorize_synthetic_shapes():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (110, 110), (0, 0, 255), thickness=-1)   # red square (BGR)
    cv2.circle(img, (180, 180), 50, (255, 0, 0), thickness=-1)            # blue circle
    svg, raw_bytes = vectorize_array(img, VCFG)
    assert validate_svg(svg) == []
    assert svg.count("<path") >= 2
    assert raw_bytes > 0 and len(svg.encode()) <= MAX_BYTES
    assert 'viewBox="0 0 512 512"' in svg  # 256px image scaled onto the 512 canvas

def test_vectorize_noisy_image_respects_cap():
    rng = np.random.default_rng(0)
    img = (rng.random((512, 512, 3)) > 0.5).astype(np.uint8) * 255
    img = cv2.resize(cv2.resize(img, (64, 64)), (512, 512), interpolation=cv2.INTER_NEAREST)
    svg, _ = vectorize_array(img, {**VCFG, "min_area": 10, "max_paths": 1000})
    assert validate_svg(svg) == []
