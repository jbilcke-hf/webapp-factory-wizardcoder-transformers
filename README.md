---
title: Webapp Factory WizardCoder Transformers
emoji: 🏭🧙
colorFrom: brown
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

A minimalist Docker project to generate apps on demand.

Ready to be used in a Hugging Face Space.

# WARNING

WARNING! This is an experimental version of the "Webapp Factory" project, and it might not work properly!

# Examples

## Local prompt examples

```
http://localhost:7860/?prompt=a%20pong%20game%20clone%20in%20HTML,%20made%20using%20the%20canvas
```
```
http://localhost:7860/?prompt=a simple html canvas game where we need to feed tadpoles controlled by an AI. The tadpoles move randomly, but when the user click inside the canvas to add some kind of food, the tadpoles will compete to eat it. Tadpole who didn't eat will die, and those who ate will reproduce.
```

## Installation

### Prerequisites

**A powerful machine is required! You need at least 24 Gb of memory!**

- Install NVM: https://github.com/nvm-sh/nvm
- Install Docker https://www.docker.com

### Python

This project relies on Python dependencies called through Pythonia.

To install those dependencies, first you should create and activate a new virtual environment for Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch
```

Note: the Dockerfile will install pytorch itself

Then install the dependencies in it:
```bash
pip install -r requirements.txt
```

### Installing the model

For this project we need to download the model in advance:

```
# go to the models directory
cd models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/WizardLM/WizardCoder-15B-V1.0

cd ..
```

Note: the Dockerfile script will do this automatically

### Building and run without Docker

```bash
nvm use
npm i
npm run start
```

### Building and running with Docker

```bash
npm run docker
```

This script is a shortcut executing the following commands:

```bash
docker build -t webapp-factory-wizardcoder-transformers .
docker run -it -p 7860:7860 webapp-factory-wizardcoder-transformers
```

