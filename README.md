# Cog Model for Flux.1-Fill-Dev, allowing both Prompt Input and Flux.1-Redux-Dev Input

[![Try a demo on Replicate](https://replicate.com/jt-street53/flux-fill-redux-dev)](https://replicate.com/jt-street53/flux-fill-redux-dev)

This is an implementation of the [Flux.1-Fill-Dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) combined with [Flux.1-Redux-Dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    sudo cog run script/download_weights

Then, you can run predictions:

    sudo cog predict -i image=@images/cartoon-man-laughing.png -i mask=@images/cartoon-man-laughing-mask.png -i reference_image=@images/cartoon-man-laughing-reference.png -i prompt="cartoon of a black woman laughing, digital art"

Push to Replicate:

    sudo cog login
    sudo cog push r8.im/<your-username>/<your-model-name>

## Example:

"modern bed with beige sheet and pillows"

![alt text](output.png)
