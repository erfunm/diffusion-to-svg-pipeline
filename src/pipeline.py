import argparse
from .diffusion_pipeline import generate_images
from .vectorize import run as vectorize_run

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    args = ap.parse_args()
    generate_images(args.config)
    vectorize_run(args.config)
