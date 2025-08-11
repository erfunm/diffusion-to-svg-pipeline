import random
from pathlib import Path
import numpy as np
import torch, yaml
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from PIL import Image

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _dtype_from_str(s: str):
    s = (s or 'auto').lower()
    if s == 'auto': return None
    if s == 'fp16': return torch.float16
    if s == 'bf16': return torch.bfloat16
    if s == 'fp32': return torch.float32
    return None

def load_pipeline(model_id: str, dtype_str: str = 'auto', use_xformers: bool = True):
    dtype = _dtype_from_str(dtype_str)
    torch_dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    if torch.cuda.is_available(): pipe = pipe.to('cuda')
    if use_xformers:
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    try: pipe.enable_attention_slicing()
    except Exception: pass
    return pipe

def generate_images(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding='utf-8'))
    dcfg, pcfg = cfg['diffusion'], cfg['paths']
    prompts = [p.strip() for p in Path(cfg['data']['prompts_file']).read_text(encoding='utf-8').splitlines() if p.strip()]
    Path(pcfg['images_dir']).mkdir(parents=True, exist_ok=True)
    _set_seed(int(dcfg.get('seed', 42)))
    pipe = load_pipeline(dcfg['model_id'], dcfg.get('dtype','auto'), dcfg.get('use_xformers', True))
    height, width = int(dcfg['height']), int(dcfg['width'])
    steps, guidance = int(dcfg['steps']), float(dcfg['guidance'])
    for i, prompt in enumerate(prompts):
        with torch.autocast('cuda', enabled=torch.cuda.is_available()):
            out = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, height=height, width=width)
        out.images[0].save(Path(pcfg['images_dir'])/f"{i:05d}.png")
    print(f"Saved images to: {pcfg['images_dir']}")
