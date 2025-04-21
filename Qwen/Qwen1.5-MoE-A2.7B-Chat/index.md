
# Qwen1.5-MoE-A2.7B-Chat
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat))



# Qwen1.5-MoE-A2.7B-Chat


## Introduction

Qwen1.5-MoE is a transformer-based MoE decoder-only language model pretrained on a large amount of data. 

For more details, please refer to our [blog post](https://qwenlm.github.io/blog/qwen-moe/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).

## Model Details
Qwen1.5-MoE employs Mixture of Experts (MoE) architecture, where the models are upcycled from dense language models. For instance, `Qwen1.5-MoE-A2.7B` is upcycled from `Qwen-1.8B`. It has 14.3B parameters in total and 2.7B activated parameters during runtime, while achieching comparable performance to `Qwen1.5-7B`, it only requires 25% of the training resources. We also observed that the inference speed is 1.74 times that of `Qwen1.5-7B`.

## Training details
We pretrained the models with a large amount of data, and we post-trained the models with both supervised finetuning and direct preference optimization. 

## Requirements
The code of Qwen1.5-MoE has been in the latest Hugging face transformers and we advise you to build from source with command `pip install git+https://github.com/huggingface/transformers`, or you might encounter the following error:
```
KeyError: 'qwen2_moe'.
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt")

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

For quantized models, we advise you to use the GPTQ correspondents, namely `Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`.


## Tips

* If you encounter code switching or other bad cases, we advise you to use our provided hyper-parameters in `generation_config.json`.
* 



## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/LICENSE) (6.7 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/README.md) (2.7 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/config.json) (873.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/configuration.json) (38.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/generation_config.json) (207.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/merges.txt) (1.6 MB)

- [model-00001-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model-00001-of-00008.safetensors) (3.7 GB)

- [model-00002-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model-00002-of-00008.safetensors) (3.7 GB)

- [model-00003-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model-00003-of-00008.safetensors) (3.7 GB)

- [model-00004-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model-00004-of-00008.safetensors) (3.7 GB)

- [model-00005-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model-00005-of-00008.safetensors) (3.7 GB)

- [model-00006-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model-00006-of-00008.safetensors) (3.7 GB)

- [model-00007-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model-00007-of-00008.safetensors) (3.7 GB)

- [model-00008-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model-00008-of-00008.safetensors) (637.5 MB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/model.safetensors.index.json) (424.9 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/tokenizer_config.json) (1.4 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-MoE-A2.7B-Chat/vocab.json) (2.6 MB)


[Back to Main](../../)