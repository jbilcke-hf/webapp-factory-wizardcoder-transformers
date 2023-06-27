FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="Hugging Face"

ENV PYTHONUNBUFFERED 1

EXPOSE 7860

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

RUN apt update

RUN apt --yes install curl

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -

RUN apt --yes install nodejs

RUN apt --yes install git git-lfs libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg

RUN git lfs install

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# prepare to install the Node app
COPY --chown=user package*.json .

RUN npm install

# then we need to install pytorch with cuda support

# If set to nothing, will install the latest version
ARG PYTORCH='2.0.1'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu118'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='TORCH_AUDIO'.*' ||  VERSION='torchaudio'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

# do we need to run this?
# RUN python3 -m pip uninstall -y tensorflow flax

# Install the rest of dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY --chown=user . .

RUN git clone https://huggingface.co/WizardLM/WizardCoder-15B-V1.0

# help Pythonia by giving it the path to Python
ENV PYTHON_BIN /usr/bin/python3

RUN python3 test.py

# CMD [ "npm", "run", "start" ]
CMD [ "npm", "run", "test" ]