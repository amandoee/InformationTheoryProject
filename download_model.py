"""Download and cache Stable Diffusion inpainting model."""

import os
import sys
import torch
from pathlib import Path

# Set Hugging Face cache directory
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
os.environ['HF_HOME'] = str(Path.home() / ".cache" / "huggingface")


try:
    from diffusers import StableDiffusionInpaintPipeline

    model = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32,
        safety_checker=None  # Disable safety checker for faster loading
    )
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
