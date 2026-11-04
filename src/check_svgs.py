"""Check every SVG in a directory against the constraints (<=10KB, allowed tags/attrs).
Exits non-zero if any file fails, so it can gate a submission.

    python -m src.check_svgs outputs/svg
"""
import argparse
import sys
from pathlib import Path
from kaggle_pkg.svg_sanitizer import validate_svg

def check_dir(svg_dir: Path) -> dict[str, list[str]]:
    return {p.name: validate_svg(p.read_text(encoding='utf-8')) for p in sorted(svg_dir.glob('*.svg'))}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('svg_dir', nargs='?', default='outputs/svg')
    results = check_dir(Path(ap.parse_args().svg_dir))
    bad = {k: v for k, v in results.items() if v}
    for name, problems in bad.items():
        print(f"FAIL {name}: {'; '.join(problems)}")
    print(f"{len(results) - len(bad)}/{len(results)} SVGs pass")
    sys.exit(1 if bad or not results else 0)
