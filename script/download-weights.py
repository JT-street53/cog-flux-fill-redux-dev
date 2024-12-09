#!/usr/bin/env python

# Run this before you deploy it on replicate
import os
import sys
import torch
from diffusers import AutoPipelineForInpainting


def download_safetensors(url: str, path: Path):
    try:
        subprocess.run(["pget", url, str(path)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download safetensors file: {e}")


# append project directory to path so predict.py can be imported
sys.path.append(".")
from predict import MODEL_NAME_FILL, MODEL_NAME_REDUX, MODEL_CACHE

# Make cache folders
if not os.path.exists(MODEL_CACHE):
    os.makedirs(MODEL_CACHE)

self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev",
    revision="refs/pr/8",
    torch_dtype=torch.bfloat16,
)
self.pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/4",
)
pipe_prior_redux.save_pretrained(MODEL_CACHE, safe_serialization=True)
pipe.save_pretrained(MODEL_CACHE, safe_serialization=True)
