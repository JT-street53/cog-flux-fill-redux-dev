FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# install basic dependencies and python
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends sudo git curl gcc \
    python3.12 python3.12-dev python3-distutils-extra \
    libgl1-mesa-glx libglib2.0-0 build-essential \ 
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# install cog
RUN curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m` && \
    chmod +x /usr/local/bin/cog

# install pip and symlink python
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -s /usr/bin/python3.12 /usr/bin/python

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip==23.3.1 && \
    pip install --no-cache-dir poetry==1.7.0 && \
    rm -rf /root/.cache/pip

# Install pip dependencies
COPY pyproject.toml /app/
RUN poetry install --no-root --no-interaction

COPY . /app/