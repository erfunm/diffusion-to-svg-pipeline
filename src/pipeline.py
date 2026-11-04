import argparse
from pathlib import Path
import yaml
from .multi_gpu import run, summarize
from .vectorize import write_submission_csv

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    prompts = [p.strip() for p in Path(cfg['data']['prompts_file']).read_text(encoding='utf-8').splitlines() if p.strip()]
    res = run(prompts, cfg, svg_dir=cfg['paths']['svg_dir'], images_dir=cfg['paths']['images_dir'])
    print(f"devices={res['devices']}", summarize(res))
    scfg = cfg.get('submission', {})
    if scfg.get('make_csv', False):
        write_submission_csv(Path(cfg['paths']['svg_dir']), Path(scfg['csv_path']))
