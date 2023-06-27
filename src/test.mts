import path from "node:path"
import { python } from "pythonia"

const transformers = await python('transformers')
const { AutoGPTQForCausalLM, BaseQuantizeConfig } = await python ('auto_gptq')

const modelName = "TheBloke/WizardCoder-15B-1.0-GPTQ"

const use_triton = false // no NVIDIA Triton Inference Server

const tokenizer = await transformers.AutoTokenizer.from_pretrained$(modelName, { use_fast: true })

const model = await AutoGPTQForCausalLM.from_quantized$(modelName, {
  use_safetensors: true,
  device: 'cuda:0',
  use_triton: false,
  quantize_config: null
})

// Prevent printing spurious transformers error when using pipeline with AutoGPTQ
await transformers.logging.set_verbosity(transformers.logging.CRITICAL)

const pipe = await transformers.pipeline$("text-generation", { model, tokenizer })

const query = "How do I sort a list in Python?"
const prompt = `<|system|>\n<|end|>\n<|user|>\n${query}<|end|>\n<|assistant|>`

//  We use a special <|end|> token with ID 49155 to denote ends of a turn
const outputs = await pipe({
  prompt,
  max_new_tokens: 256,
  do_sample: true,
  temperature: 0.2,
  top_k: 50,
  top_p: 0.95,
  eos_token_id: 49155
})

// You can sort a list in Python by using the sort() method. Here's an example:\n\n```\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nnumbers.sort()\nprint(numbers)\n```\n\nThis will sort the list in place and print the sorted list.
console.log(outputs[0]['generated_text'])
