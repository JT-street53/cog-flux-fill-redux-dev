#!/usr/bin/env python

# Run this before you deploy it on replicate
import os
import sys
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import FluxPriorReduxPipeline, FluxFillPipeline


# append project directory to path so predict.py can be imported
sys.path.append(".")
from predict import MODEL_NAME_FILL, MODEL_NAME_REDUX, MODEL_CACHE

# Make cache folders
if not os.path.exists(MODEL_CACHE):
    os.makedirs(MODEL_CACHE)

# Login to Hugging Face
load_dotenv()
login(token=os.environ["HUGGINGFACE_TOKEN"])

# Download Flux Prior Redux
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    MODEL_NAME_REDUX,
    torch_dtype=torch.bfloat16,
)
pipe_prior_redux.save_pretrained(MODEL_CACHE, safe_serialization=True)

# Download Flux Fill
pipe = FluxFillPipeline.from_pretrained(
    MODEL_NAME_FILL,
    torch_dtype=torch.bfloat16,
)
pipe.save_pretrained(MODEL_CACHE, safe_serialization=True)
