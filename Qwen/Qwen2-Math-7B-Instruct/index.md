
# Qwen2-Math-7B-Instruct
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2-Math-7B-Instruct))




# Qwen2-Math-7B-Instruct

<h2 align="center">
  <b>
    <span style="color: red;">
      🚨 Temporarily this model mainly supports English. We will release bilingual (English & Chinese) models soon!
    </span>
  </b>
</h2>

## Introduction

Over the past year, we have dedicated significant effort to researching and enhancing the reasoning capabilities of large language models, with a particular focus on their ability to solve arithmetic and mathematical problems. Today, we are delighted to introduce a serise of math-specific large language models of our Qwen2 series,  Qwen2-Math and Qwen2-Math-Instruct-1.5B/7B/72B. Qwen2-Math is a series of specialized math language models built upon the Qwen2 LLMs, which significantly outperforms the mathematical capabilities of open-source models and even closed-source models (e.g., GPT4o). We hope that Qwen2-Math can contribute to the scientific community for solving advanced mathematical problems that require complex, multi-step logical reasoning.


## Model Details


For more details, please refer to our [blog post](https://qwenlm.github.io/blog/qwen2-math/) and [GitHub repo](https://github.com/QwenLM/Qwen2-Math).


## Requirements
* `transformers>=4.40.0` for Qwen2-Math models. The latest version is recommended.

<h2 align="center">
  <b>
    <span style="color: red;">
      🚨 This is a must because `transformers` integrated Qwen2 codes since `4.37.0`
    </span>
  </b>
</h2>

For requirements on GPU memory and the respective throughput, see similar results of Qwen2 [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Quick Start

**Qwen2-Math-7B-Instruct** is an instruction model for chatting;

**Qwen2-Math-7B** is a base model typically used for completion and few-shot inference, serving as a better starting point for fine-tuning.
 

### 🤗 Hugging Face Transformers

Qwen2-Math can be deployed and inferred in the same way as [Qwen2](https://github.com/QwenLM/Qwen2). Here we show a code snippet to show you how to use the chat model with `transformers`:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "qwen/Qwen2-Math-7B-Instruct"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    
    
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pd")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### 🤖 ModelScope
We strongly advise users, especially those in mainland China, to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.


## Citation

If you find our work helpful, feel free to give us a citation.

```
@article{yang2024qwen2,
  title={Qwen2 technical report},
  author={Yang, An and Yang, Baosong and Hui, Binyuan and Zheng, Bo and Yu, Bowen and Zhou, Chang and Li, Chengpeng and Li, Chengyuan and Liu, Dayiheng and Huang, Fei and others},
  journal={arXiv preprint arXiv:2407.10671},
  year={2024}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/README.md) (3.6 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/config.json) (674.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/configuration.json) (73.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/generation_config.json) (197.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/merges.txt) (1.6 MB)

- [model-00001-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/model-00001-of-00004.safetensors) (3.7 GB)

- [model-00002-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/model-00002-of-00004.safetensors) (3.6 GB)

- [model-00003-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/model-00003-of-00004.safetensors) (3.6 GB)

- [model-00004-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/model-00004-of-00004.safetensors) (3.3 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/model.safetensors.index.json) (27.1 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/tokenizer_config.json) (1.3 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-7B-Instruct/vocab.json) (2.6 MB)


[Back to Main](../../)