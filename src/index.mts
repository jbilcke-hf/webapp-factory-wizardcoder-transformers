import os from "node:os"
import path from "node:path"
import express from "express"
import { python } from "pythonia"

import { daisy } from "./daisy.mts"
import { alpine } from "./alpine.mts"

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


// define the CSS and JS dependencies
const css = [
  "/css/daisyui@2.6.0.css",
].map(item => `<link href="${item}" rel="stylesheet" type="text/css"/>`)
.join("")

const script = [
  "/js/alpinejs@3.12.2.js",
  "/js/tailwindcss@3.3.2.js"
].map(item => `<script src="${item}"></script>`)
.join("")

const app = express()
const port = 7860

const minPromptSize = 16 // if you change this, you will need to also change in public/index.html
const timeoutInSec = 60 * 60

console.log("timeout set to 60 minutes")

app.use(express.static("public"))
 
const maxParallelRequests = 1

const pending: {
  total: number;
  queue: string[];
} = {
  total: 0,
  queue: [],
}
 
const endRequest = (id: string, reason: string) => {
  if (!id || !pending.queue.includes(id)) {
    return
  }
  
  pending.queue = pending.queue.filter(i => i !== id)
  console.log(`request ${id} ended (${reason})`)
}

// we need to exit the open Python process or else it will keep running in the background
process.on('SIGINT', () => {
  try {
    (python as any).exit()
  } catch (err) {
    // exiting Pythonia can get a bit messy: try/catch or not,
    // you *will* see warnings and tracebacks in the console
  }
  process.exit(0)
})

app.get("/debug", (req, res) => {
  res.write(JSON.stringify({
    nbTotal: pending.total,
    nbPending: pending.queue.length,
    queue: pending.queue,
  }))
  res.end()
})

app.get("/app", async (req, res) => {

  if (`${req.query.prompt}`.length < minPromptSize) {
    res.write(`prompt too short, please enter at least ${minPromptSize} characters`)
    res.end()
    return
  }
  // naive implementation: we say we are out of capacity
  if (pending.queue.length >= maxParallelRequests) {
    res.write("sorry, max nb of parallel requests reached")
    res.end()
    return
  }
  // alternative approach: kill old queries
  // while (pending.queue.length > maxParallelRequests) {
  //   endRequest(pending.queue[0], 'max nb of parallel request reached')
  // }

  const id = `${pending.total++}`
  console.log(`new request ${id}`)

  pending.queue.push(id)

  const prefix = `<html><head>${css}${script}`
  res.write(prefix)

  req.on("close", function() {
    // console.log("browser asked to close the stream for some reason.. let's ignore!")
    endRequest(id, "browser asked to end the connection")
  })

  // for testing we kill after some delay
  setTimeout(() => {
    endRequest(id, `timed out after ${timeoutInSec}s`)
  }, timeoutInSec * 1000)


  const finalPrompt = `# Context
Generate this webapp: ${req.query.prompt}.
# Documentation
${daisy}
# Guidelines
- Do not write a tutorial! This is a web app!
- Never repeat the instruction, instead directly write the final code within a script tag
- Use a color scheme consistent with the brief and theme
- You need to use Tailwind CSS and DaisyUI for the UI, pure vanilla JS and AlpineJS for the JS.
- All the JS code will be written directly inside the page, using <script type="text/javascript">...</script>
- You MUST use English, not Latin! (I repeat: do NOT write lorem ipsum!)
- No need to write code comments, and try to make the code compact (short function names etc)
- Use a central layout by wrapping everything in a \`<div class="flex flex-col justify-center">\`
# HTML Code of the final app:
${prefix}`

      
  try {

    // be careful: if you input a prompt which is too large, you may experience a timeout
    // const inputTokens = await llm.tokenize(finalPrompt)
    console.log("initializing the generator (may take 30s or more)")

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
    const tmp = outputs[0]['generated_text']

    process.stdout.write(tmp)
    res.write(tmp)
    /*
    const inputTokens = await tokenizer.tokenize("def fibbinacci(")
    console.log('inputTokens:', inputTokens)
    const tmp = await llm.generate_tokens$({
      tokens: inputTokens,
      max_length: 64,
      include_prompt_in_result: false,
    })
    process.stdout.write(tmp)
    res.write(tmp)
    console.log("generator initialized, beginning token streaming..")
    for await (const token of generator) {
      if (!pending.queue.includes(id)) {
        break
      }
      const tmp = await tokenizer.detokenize(token)
      process.stdout.write(tmp)
      res.write(tmp)
    }
    */

    endRequest(id, `normal end of the LLM stream for request ${id}`)
  } catch (e) {
    endRequest(id, `premature end of the LLM stream for request ${id} (${e})`)
  } 

  try {
    res.end()
  } catch (err) {
    console.log(`couldn't end the HTTP stream for request ${id} (${err})`)
  }
  
})

app.listen(port, () => { console.log(`Open http://localhost:${port}/?prompt=a%20pong%20game%20clone%20in%20HTML,%20made%20using%20the%20canvas`) })

