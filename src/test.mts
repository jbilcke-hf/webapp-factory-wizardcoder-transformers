import path from "node:path"
import { python } from "pythonia"

const torch = await python('torch')
const sys = await python('sys')
const { AutoTokenizer, AutoModelForCausalLM } = await python ('transformers')

let device = await torch.cuda.is_available() ? 'cuda' : 'cpu'

try {
  if (await torch.backends.mps.is_available()) {
    device = 'mps'
  }
} catch (_err) {
}

console.log('device: ' + device)

const baseModel = "WizardLM/WizardCoder-15B-V1.0"
const load8bit = false

let model = null

console.log("loading tokenizer..")
const tokenizer = AutoTokenizer.from_pretrained(baseModel)
if (device == "cuda") {
  model = await AutoModelForCausalLM.from_pretrained$(
    baseModel, {
    load_in_8bit: load8bit,
    torch_dtype: torch.float16,
    device_map: "auto",
  })
} else if (device == "mps") {
  model = await AutoModelForCausalLM.from_pretrained(
    baseModel, {
    device_map: {"": device},
    torch_dtype: torch.float16,
  })
}

console.log("loaded tokenizer")

model.config.pad_token_id = tokenizer.pad_token_id
if (!load8bit) {
  await model.half()
}

console.log("calling model.eval()")
await model.eval()

if (torch.__version__ >= "2" && sys.platform != "win32") {
  console.log("calling torch.compile(model)")
  model = await torch.compile(model)
}

console.log("calling evaluate..")

const prompt = `Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Write a short summary about AI.

### Response:`
const inputs = await tokenizer(
  prompt, {
  return_tensors: "pt",
  truncation: true,
  padding: true
})
const input_ids = await inputs["input_ids"].to(device)


await torch.no_grad()
const generated = await model.generate(input_ids)
const s = generated
const output = await tokenizer.decode$(generated[0], { skip_special_tokens: true })
console.log('output: ', output)