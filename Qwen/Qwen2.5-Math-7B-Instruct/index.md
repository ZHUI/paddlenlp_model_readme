
# Qwen2.5-Math-7B-Instruct
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct))

---
base_model: Qwen/Qwen2.5-Math-7B
language:
- en
pipeline_tag: text-generation
tags:
- chat
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct/blob/main/LICENSE
---


# Qwen2.5-Math-7B-Instruct

> [!Warning]
> <div align="center">
> <b>
> 🚨 Qwen2.5-Math mainly supports solving English and Chinese math problems through CoT and TIR. We do not recommend using this series of models for other tasks.
> </b>
> </div>

## Introduction

In August 2024, we released the first series of mathematical LLMs - [Qwen2-Math](https://qwenlm.github.io/blog/qwen2-math/) - of our Qwen family. A month later, we have upgraded it and open-sourced **Qwen2.5-Math** series, including base models **Qwen2.5-Math-1.5B/7B/72B**, instruction-tuned models **Qwen2.5-Math-1.5B/7B/72B-Instruct**, and mathematical reward model **Qwen2.5-Math-RM-72B**. 
                                                                              
Unlike Qwen2-Math series which only supports using Chain-of-Thught (CoT) to solve English math problems, Qwen2.5-Math series is expanded to support using both CoT and Tool-integrated Reasoning (TIR) to solve math problems in both Chinese and English. The Qwen2.5-Math series models have achieved significant performance improvements compared to the Qwen2-Math series models on the Chinese and English mathematics benchmarks with CoT. 

![![](http://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5/qwen2.5-math-pipeline.jpeg)

While CoT plays a vital role in enhancing the reasoning capabilities of LLMs, it faces challenges in achieving computational accuracy and handling complex mathematical or algorithmic reasoning tasks, such as finding the roots of a quadratic equation or computing the eigenvalues of a matrix. TIR can further improve the model's proficiency in precise computation, symbolic manipulation, and algorithmic manipulation. Qwen2.5-Math-1.5B/7B/72B-Instruct achieve 79.7, 85.3, and 87.8 respectively on the MATH benchmark using TIR. 

## Model Details


For more details, please refer to our [blog post](https://qwenlm.github.io/blog/qwen2.5-math/) and [GitHub repo](https://github.com/QwenLM/Qwen2.5-Math).


## Requirements
* `transformers>=4.37.0` for Qwen2.5-Math models. The latest version is recommended.

> [!Warning]
> <div align="center">
> <b>
> 🚨 This is a must because <code>transformers</code> integrated Qwen2 codes since <code>4.37.0</code>.
> </b>
> </div>

For requirements on GPU memory and the respective throughput, see similar results of Qwen2 [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Quick Start

> [!Important]
>
> **Qwen2.5-Math-7B-Instruct** is an instruction model for chatting;
>
> **Qwen2.5-Math-7B** is a base model typically used for completion and few-shot inference, serving as a better starting point for fine-tuning.
> 

### 🤗 Hugging Face Transformers

Qwen2.5-Math can be deployed and infered in the same way as [Qwen2.5](https://github.com/QwenLM/Qwen2.5). Here we show a code snippet to show you how to use the chat model with `transformers`:

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "qwen/Qwen2.5-Math-7B-Instruct"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
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
)[0]
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### 🤖 ModelScope
We strongly advise users especially those in mainland China to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.


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

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/LICENSE) (11.1 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/README.md) (4.6 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/config.json) (652.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/configuration.json) (2.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/generation_config.json) (161.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/merges.txt) (1.6 MB)

- [model-00001-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/model-00001-of-00004.safetensors) (3.7 GB)

- [model-00002-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/model-00002-of-00004.safetensors) (3.6 GB)

- [model-00003-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/model-00003-of-00004.safetensors) (3.6 GB)

- [model-00004-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/model-00004-of-00004.safetensors) (3.3 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/model.safetensors.index.json) (27.1 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/tokenizer_config.json) (7.1 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-7B-Instruct/vocab.json) (2.6 MB)


[Back to Main](../../)