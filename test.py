import sys
import os
import fire
import torch
import transformers
import json
import jsonlines

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

print("device: " + device)

base_model = "./models/WizardCoder-15B-V1.0"
load_8bit = False

tokenizer = AutoTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
elif device == "mps":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
model.config.pad_token_id = tokenizer.pad_token_id
if not load_8bit:
    model.half()

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
  model = torch.compile(model)
