
# Mixtral-8x7B-Instruct-v0.1
---


## README([From Huggingface](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1))

---
language:
- fr
- it
- de
- es
- en
license: apache-2.0
base_model: mistralai/Mixtral-8x7B-v0.1
inference:
  parameters:
    temperature: 0.5
widget:
- messages:
  - role: user
    content: What is your favorite condiment?

extra_gated_description: If you want to learn more about how we process your personal data, please read our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
---
# Model Card for Mixtral-8x7B

### Tokenization with `mistral-common`

```py
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
 
mistral_models_path = "MISTRAL_MODELS_PATH"
 
tokenizer = MistralTokenizer.v1()
 
completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
 
tokens = tokenizer.encode_chat_completion(completion_request).tokens
```
 
## Inference with `mistral_inference`
 
 ```py
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
 
model = Transformer.from_folder(mistral_models_path)
out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)

result = tokenizer.decode(out_tokens[0])

print(result)
```

## Inference with hugging face `transformers`
 
```py
from paddlenlp.transformers import AutoModelForCausalLM
 
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model.to("cuda")
 
generated_ids = model.generate(tokens, max_new_tokens=1000, do_sample=True)

# decode with mistral tokenizer
result = tokenizer.decode(generated_ids[0].tolist())
print(result)
```

> [!TIP]
> PRs to correct the transformers tokenizer so that it gives 1-to-1 the same results as the mistral-common reference implementation are very welcome!
     
        
---
The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mixtral-8x7B outperforms Llama 2 70B on most benchmarks we tested.

For full details of this model please read our [release blog post](https://mistral.ai/news/mixtral-of-experts/).

## Warning
This repo contains weights that are compatible with [vLLM](https://github.com/vllm-project/vllm) serving of the model as well as Hugging Face [transformers](https://github.com/huggingface/transformers) library. It is based on the original Mixtral [torrent release](magnet:?xt=urn:btih:5546272da9065eddeb6fcd7ffddeef5b75be79a7&dn=mixtral-8x7b-32kseqlen&tr=udp%3A%2F%http://2Fopentracker.i2p.rocks%3A6969%2Fannounce&tr=http%3A%2F%http://2Ftracker.openbittorrent.com%3A80%2Fannounce), but the file format and parameter names are different. Please note that model cannot (yet) be instantiated with HF.

## Instruction format

This format must be strictly respected, otherwise the model will generate sub-optimal outputs.

The template used to build a prompt for the Instruct model is defined as follows:
```
<s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
```
Note that `<s>` and `</s>` are special tokens for beginning of string (BOS) and end of string (EOS) while [INST] and [/INST] are regular strings.

As reference, here is the pseudo-code used to tokenize instructions during fine-tuning:
```python
def tokenize(text):
    return tok.encode(text, add_special_tokens=False)

[BOS_ID] + 
tokenize("[INST]") + tokenize(USER_MESSAGE_1) + tokenize("[/INST]") +
tokenize(BOT_MESSAGE_1) + [EOS_ID] +
…
tokenize("[INST]") + tokenize(USER_MESSAGE_N) + tokenize("[/INST]") +
tokenize(BOT_MESSAGE_N) + [EOS_ID]
```

In the pseudo-code above, note that the `tokenize` method should not add a BOS or EOS token automatically, but should add a prefix space. 

In the Transformers library, one can use [chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating) which make sure the right format is applied.

## Run the model

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

By default, transformers will load the model in full precision. Therefore you might be interested to further reduce down the memory requirements to run the model through the optimizations we offer in HF ecosystem:

### In half-precision

Note `float16` precision only works on GPU devices

<details>
<summary> Click to expand </summary>

```diff
+ import torch
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

### Lower precision using (8-bit & 4-bit) using `bitsandbytes`

<details>
<summary> Click to expand </summary>

```diff
+ import torch
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

text = "Hello my name is"
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

### Load the model with Flash Attention 2

<details>
<summary> Click to expand </summary>

```diff
+ import torch
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=True, device_map="auto")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

## Limitations

The Mixtral-8x7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. 
It does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to
make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.

# The Mistral AI Team
Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Louis Ternon, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/README.md) (8.5 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/config.json) (978.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/generation_config.json) (167.0 B)

- [model-00001-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00001-of-00019.safetensors) (4.6 GB)

- [model-00002-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00002-of-00019.safetensors) (4.6 GB)

- [model-00003-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00003-of-00019.safetensors) (4.6 GB)

- [model-00004-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00004-of-00019.safetensors) (4.6 GB)

- [model-00005-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00005-of-00019.safetensors) (4.6 GB)

- [model-00006-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00006-of-00019.safetensors) (4.6 GB)

- [model-00007-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00007-of-00019.safetensors) (4.6 GB)

- [model-00008-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00008-of-00019.safetensors) (4.6 GB)

- [model-00009-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00009-of-00019.safetensors) (4.6 GB)

- [model-00010-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00010-of-00019.safetensors) (4.6 GB)

- [model-00011-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00011-of-00019.safetensors) (4.6 GB)

- [model-00012-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00012-of-00019.safetensors) (4.6 GB)

- [model-00013-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00013-of-00019.safetensors) (4.6 GB)

- [model-00014-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00014-of-00019.safetensors) (4.6 GB)

- [model-00015-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00015-of-00019.safetensors) (4.6 GB)

- [model-00016-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00016-of-00019.safetensors) (4.6 GB)

- [model-00017-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00017-of-00019.safetensors) (4.6 GB)

- [model-00018-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00018-of-00019.safetensors) (4.6 GB)

- [model-00019-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00019-of-00019.safetensors) (3.9 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/model.safetensors.index.json) (92.4 KB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/sentencepiece.bpe.model) (481.9 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/special_tokens_map.json) (72.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/tokenizer.json) (1.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-Instruct-v0.1/tokenizer_config.json) (1.4 KB)


[Back to Main](../../)