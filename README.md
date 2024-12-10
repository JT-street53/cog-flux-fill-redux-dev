# Cog Model for Flux.1-Fill-Dev, allowing both Prompt Input and Flux.1-Redux-Dev Input

[![Try a demo on Replicate](https://replicate.com/lucataco/sdxl-inpainting/badge)](https://replicate.com/lucataco/sdxl-inpainting)

This is an implementation of the [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    sudo cog run script/download-weights.py

Then, you can run predictions:

    sudo cog predict -i image=@images/cartoon-man-laughing.png -i mask=@images/cartoon-man-laughing-mask.png -i reference_image=@images/cartoon-man-laughing.png -i prompt="cartoon of a black woman laughing, digital art"

## Example:

"modern bed with beige sheet and pillows"

![alt text](output.png)
