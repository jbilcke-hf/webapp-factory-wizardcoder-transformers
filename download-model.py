import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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


def evaluate(instruction, tokenizer, model):
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    
    with torch.no_grad():
        generation_output = model.generate(input_ids)
    s = generation_output
    output = tokenizer.decode(s[0], skip_special_tokens=True)
    return output.split("### Response:")[1].strip()

base_model = "WizardLM/WizardCoder-15B-V1.0"
load_8bit = False

print("loading tokenizer..")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

print("loaded tokenizer")
model.config.pad_token_id = tokenizer.pad_token_id
if not load_8bit:
    model.half()

print("calling model.eval()")
model.eval()

if torch.__version__ >= "2" and sys.platform != "win32":
    print("calling torch.compile(model)")
    model = torch.compile(model)

instruction = "Write a short summary about AI."
print("calling evaluate..")
result = evaluate(instruction, tokenizer, model)

print("result: ")
print(result)
