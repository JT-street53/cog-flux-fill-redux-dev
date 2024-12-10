#!/usr/bin/env python

import os
import sys
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import FluxPriorReduxPipeline, FluxFillPipeline

MODEL_NAME_FILL = "black-forest-labs/FLUX.1-Fill-dev"
MODEL_NAME_REDUX = "black-forest-labs/FLUX.1-Redux-dev"
MODEL_CACHE = "checkpoints"


def download_weights():
    """Download the model weights from Hugging Face and save them locally."""
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


if __name__ == "__main__":
    download_weights()
