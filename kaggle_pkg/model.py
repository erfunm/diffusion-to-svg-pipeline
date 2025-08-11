import os, math, re
from typing import Any
from .svg_sanitizer import sanitize_svg

class Model:
    """Kaggle Package interface.
    Must define a class `Model` with a `predict(prompt: str) -> str` method that returns SVG code.
    This implementation:
    - Uses lightweight heuristics to compose shapes from keywords (fast, deterministic).
    - Optionally, if DIFFUSION_TO_SVG=1, will try a raster->vector pipeline (for local use only).
    """

    def __init__(self, config: dict | None = None):
        self.width = (config or {}).get("width", 512)
        self.height = (config or {}).get("height", 512)
        self.diffusion_enabled = os.getenv("DIFFUSION_TO_SVG", "0") == "1"

    # --- Simple heuristic renderer (keyword → shapes) ---
    def _heuristic_svg(self, prompt: str) -> str:
        p = prompt.lower()
        w, h = self.width, self.height
        elems = []

        def clamp01(x): return max(0, min(1, x))

        # Colors
        color = "#000000"
        if "red" in p: color = "#e53935"
        elif "blue" in p: color = "#1e88e5"
        elif "green" in p: color = "#43a047"
        elif "yellow" in p: color = "#fdd835"
        elif "black" in p: color = "#000000"
        elif "white" in p: color = "#ffffff"

        # Shapes by keywords
        if "circle" in p or "dot" in p:
            cx, cy, r = w*0.5, h*0.5, min(w,h)*0.25
            d = f"M {cx-r} {cy} A {r} {r} 0 1 0 {cx+r} {cy} A {r} {r} 0 1 0 {cx-r} {cy} Z"
            elems.append(f"<path d=\"{d}\" fill=\"{color}\"/>")
        if "square" in p or "box" in p:
            x0, y0, s = w*0.25, h*0.25, min(w,h)*0.5
            d = f"M {x0} {y0} L {x0+s} {y0} L {x0+s} {y0+s} L {x0} {y0+s} Z"
            elems.append(f"<path d=\"{d}\" fill=\"{color}\"/>")
        if "triangle" in p:
            x0, y0 = w*0.5, h*0.25
            x1, y1 = w*0.25, h*0.75
            x2, y2 = w*0.75, h*0.75
            d = f"M {x0} {y0} L {x1} {y1} L {x2} {y2} Z"
            elems.append(f"<path d=\"{d}\" fill=\"{color}\"/>")
        if "star" in p:
            cx, cy, R, r = w*0.5, h*0.5, min(w,h)*0.35, min(w,h)*0.15
            pts = []
            for i in range(10):
                a = i * math.pi/5
                rad = R if i % 2 == 0 else r
                pts.append((cx + rad*math.cos(a), cy + rad*math.sin(a)))
            d = "M " + " ".join([f"L {x} {y}" for x,y in pts])[2:] + " Z"
            elems.append(f"<path d=\"{d}\" fill=\"{color}\"/>")

        # Fallback: simple icon
        if not elems:
            d = f"M {w*0.2} {h*0.5} L {w*0.8} {h*0.5} L {w*0.5} {h*0.2} Z"
            elems.append(f"<path d=\"{d}\" fill=\"{color}\"/>")

        svg = f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">{''.join(elems)}</svg>"
        return sanitize_svg(svg, width=w, height=h)

    # --- Optional heavier pipeline (local) ---
    def _diffusion_to_svg(self, prompt: str) -> str:
        from PIL import Image
        import numpy as np, cv2, svgwrite, torch
        from diffusers import AutoPipelineForText2Image
        # 1) raster
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available(): pipe = pipe.to("cuda")
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            out = pipe(prompt, num_inference_steps=4, guidance_scale=0.0, height=self.height, width=self.width)
        img = out.images[0]
        # 2) vectorize (very light)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=lambda c: -cv2.contourArea(c))[:128]
        paths = []
        for c in cnts:
            if cv2.contourArea(c) < 150: continue
            approx = cv2.approxPolyDP(c, 2.0, True).reshape(-1,2)
            if len(approx) < 3: continue
            d = f"M {approx[0][0]} {approx[0][1]} " + " ".join([f"L {x} {y}" for x,y in approx[1:]]) + " Z"
            paths.append(f"<path d=\"{d}\" fill=\"#000000\"/>")
        svg = f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{self.width}\" height=\"{self.height}\" viewBox=\"0 0 {self.width} {self.height}\">{''.join(paths)}</svg>"
        return sanitize_svg(svg, width=self.width, height=self.height)

    def predict(self, prompt: str) -> str:
        if self.diffusion_enabled:
            try:
                return self._diffusion_to_svg(prompt)
            except Exception:
                # Fallback to heuristic
                return self._heuristic_svg(prompt)
        return self._heuristic_svg(prompt)

if __name__ == "__main__":
    # Simple CLI for local testing
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="a blue circle icon") 
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()
    m = Model({"width": args.width, "height": args.height})
    print(m.predict(args.prompt))
