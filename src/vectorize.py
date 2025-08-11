from pathlib import Path
import yaml, cv2, numpy as np, svgwrite, csv

def rgb_to_hex(color):
    r, g, b = [int(x) for x in color]
    return f"#{r:02x}{g:02x}{b:02x}"

def contour_mean_color(img_bgr, contour):
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    mean_bgr = cv2.mean(img_bgr, mask=mask)[:3]
    b, g, r = mean_bgr
    return (r, g, b)

def vectorize_image(png_path: Path, svg_dir: Path, vcfg: dict):
    img_bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, vcfg['canny_low'], vcfg['canny_high'])
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < vcfg['min_area']: continue
        approx = cv2.approxPolyDP(c, float(vcfg['approx_epsilon']), True).reshape(-1,2)
        if len(approx) < 3: continue
        filtered.append(approx)
    filtered.sort(key=lambda pts: -cv2.contourArea(pts.reshape(-1,1,2)))
    filtered = filtered[: int(vcfg['max_paths']) ]

    W, H = vcfg['canvas_size']
    out_file = svg_dir / (png_path.stem + '.svg')
    dwg = svgwrite.Drawing(str(out_file), size=(W, H))
    dwg.viewbox(0, 0, w, h)

    use_fill = bool(vcfg.get('use_fill', True))
    use_stroke = bool(vcfg.get('use_stroke', False))

    for pts in filtered:
        d = f"M {pts[0][0]} {pts[0][1]} " + " ".join([f"L {x} {y}" for x,y in pts[1:]]) + " Z"
        color = contour_mean_color(img_bgr, pts.reshape(-1,1,2))
        fill = rgb_to_hex(color) if use_fill else "none"
        stroke = "#000000" if use_stroke else "none"
        dwg.add(dwg.path(d=d, fill=fill, stroke=stroke))

    dwg.save()

def run(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding='utf-8'))
    images_dir = Path(cfg['paths']['images_dir'])
    svg_dir = Path(cfg['paths']['svg_dir'])
    svg_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(images_dir.glob('*.png'))
    if not pngs:
        print(f"No PNGs in {images_dir}. Run generation first."); return
    for p in pngs: vectorize_image(p, svg_dir, cfg['vectorize'])
    print(f"Saved SVGs to: {svg_dir}")

    scfg = cfg.get('submission', {})
    if scfg.get('make_csv', False):
        csv_path = Path(scfg['csv_path']); csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['id','svg'])
            for s in sorted(svg_dir.glob('*.svg')): w.writerow([s.stem, s.read_text(encoding='utf-8')])
        print(f"Wrote CSV: {csv_path}")
