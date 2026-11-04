import argparse
import random
import warnings
from pathlib import Path
import numpy as np
import torch, yaml

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    """'auto' -> fp16 on CUDA, fp32 on CPU. Half precision on CPU falls back to fp32."""
    s = (dtype_str or 'auto').lower()
    on_gpu = device.startswith('cuda')
    if s == 'auto':
        return torch.float16 if on_gpu else torch.float32
    if s not in _DTYPES:
        raise ValueError(f"dtype must be one of auto/{'/'.join(_DTYPES)}, got {dtype_str!r}")
    if not on_gpu and s != 'fp32':
        warnings.warn(f"{s} requested on {device}; using fp32")
        return torch.float32
    return _DTYPES[s]

def load_pipeline(model_id: str, device: str = 'cuda', dtype_str: str = 'auto',
                  use_xformers: bool = True, attention_slicing: bool = True):
    """Load a text-to-image pipeline on `device`. Returns (pipe, info) where info records
    what was actually applied (not just requested).

    Note: in diffusers, xFormers and attention slicing each replace the attention processor,
    so they do not stack. Slicing is applied first and xFormers second, so xFormers wins when
    it is available and slicing remains as the memory-saving fallback when it is not.
    """
    from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
    torch_dtype = resolve_dtype(dtype_str, device)
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if attention_slicing:
        pipe.enable_attention_slicing()
    xformers_active = False
    if use_xformers and device.startswith('cuda'):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            xformers_active = True
        except Exception as e:
            warnings.warn(f"xFormers unavailable on {device}, continuing without it: {e}")
    procs = sorted({type(p).__name__ for p in pipe.unet.attn_processors.values()})
    info = {'device': device, 'dtype': str(torch_dtype).replace('torch.', ''),
            'xformers_active': xformers_active, 'attn_processor': '+'.join(procs)}
    return pipe, info

def generate_images(cfg_path: str):
    """Single-device generation (kept for the Colab notebook). See src/multi_gpu.py for multi-GPU."""
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding='utf-8'))
    dcfg, pcfg = cfg['diffusion'], cfg['paths']
    prompts = [p.strip() for p in Path(cfg['data']['prompts_file']).read_text(encoding='utf-8').splitlines() if p.strip()]
    Path(pcfg['images_dir']).mkdir(parents=True, exist_ok=True)
    _set_seed(int(dcfg.get('seed', 42)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe, _ = load_pipeline(dcfg['model_id'], device, dcfg.get('dtype', 'auto'),
                            dcfg.get('use_xformers', True), dcfg.get('attention_slicing', True))
    height, width = int(dcfg['height']), int(dcfg['width'])
    steps, guidance = int(dcfg['steps']), float(dcfg['guidance'])
    for i, prompt in enumerate(prompts):
        out = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, height=height, width=width)
        out.images[0].save(Path(pcfg['images_dir'])/f"{i:05d}.png")
    print(f"Saved images to: {pcfg['images_dir']}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    generate_images(ap.parse_args().config)
