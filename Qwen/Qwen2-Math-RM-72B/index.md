
# Qwen2-Math-RM-72B
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2-Math-RM-72B))




# Qwen2-Math-RM-72B

## Introduction
Qwen2-Math-RM-72B is specifically designed to guide the Qwen2-Math model throughout the training process by offering more granular feedback on the quality of reasoning and intermediate steps, ultimately facilitating more robust model improvements.


Key Highlights:

- Model Training Guide:
  - Training Data Enhancement: Employs a data selection process via reward model scoring combined with Rejection Sampling to incrementally enhance the quality of responses
  - Reinforcement Learning Training: Integrates seamlessly into the reinforcement learning training and provide effective reward signal, further improving model performance.

- Inference Boosting:
  - Best of N: By leveraging a combination of response sampling and Best-of-N strategies, we choose the response of top score judged by reward model, yielding better results with spending more inference time. For example, Qwen2-Math-1.5B-Instruct obtains 79.9 on MATH in RM@8 setting and even surpasses the performance of Qwen2-Math-7B-Instruct 75.1 with greedy decoding.
  - Comparasion with majority voting (Maj@N): RM@N scores are substantially better than Maj@N scores aross almost all benchmarks and models.


## Model Details

For more details, please refer to our [blog post](https://qwenlm.github.io/blog/qwen2-math/) and [GitHub repo](https://github.com/QwenLM/Qwen2-Math).


## Requirements
* `transformers>=4.40.0` for Qwen2-Math models. The latest version is recommended.

> [!Warning]
> <div align="center">
> <b>
> 🚨 This is a must because `transformers` integrated Qwen2 codes since `4.37.0`.
> </b>
> </div>

For requirements on GPU memory and the respective throughput, see similar results of Qwen2 [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Quick Start

> [!Important]
>
> **Qwen2-Math-RM-72B** is a reward model typically used for offering feedback on the quality of reasoning and intermediate steps, serving in Rejection Sampling, reinforcement learning training and RM@N.

### 🤗 Hugging Face Transformers

Here we show a code snippet to show you how to use the Qwen2-Math-RM-72B with `transformers`:

```python
import torch
from paddlenlp.transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen2-Math-RM-72B"
device = "auto" # the device to load the model onto

model = AutoModel.from_pretrained(
    model_name, 
    device_map=device, 
    dtype=paddle.bfloat16,
    trust_remote_code=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

chat = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "Suppose that a certain software product has a mean time between failures of 10,000 hours and has a mean time to repair of 20 hours. If the product is used by 100 customers, what is its availability?\nAnswer Choices: (A) 80% (B) 90% (C) 98% (D) 99.80%\nPlease reason step by step, and put your final answer within \\boxed{}."},
    {"role": "assistant", "content": "To find the availability of the software product, we'll use the formula:\n\n\\[ \\text{ availability} = \\frac{\\text{Mean Time Between Failures (MTBF)}}{\\text{Mean Time Between Failures (MTBF) + Mean Time To Repair (MTTR)}} \\]\n\nGiven:\n- MTBF = 10,000 hours\n- MTTR = 20 hours\n\nLet's plug these values into the formula:\n\n\\[ \\text{availability} = \\frac{10,000}{10,000 + 20} = \\frac{10,000}{10,020} \\]\n\nTo simplify this fraction, we can divide both the numerator and the denominator by 10,000:\n\n\\[ \\text{availability} = \\frac{10,000 \\div 10,000}{10,020 \\div 10,000} = \\frac{1}{1.002} \\]\n\nTo express this as a percentage, we can calculate the decimal value of the fraction and then multiply by 100:\n\n\\[ \\text{availability} \\approx 0.998002 \\times 100 = 99.80\\% \\]\n\nTherefore, the availability of the software product is approximately 99.80%.\n\nThe correct answer is \\boxed{D}"}
] # 1.7578

conversation_str = tokenizer.apply_chat_template(
    chat, 
    tokenize=False, 
    add_generation_prompt=False
)

input_ids = tokenizer.encode(
    conversation_str, 
    return_tensors="pd", 
    add_special_tokens=False
)

outputs = model(input_ids=input_ids)
print(outputs[0])
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

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/LICENSE) (6.7 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/README.md) (4.9 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/config.json) (813.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/configuration.json) (48.0 B)

- [configuration_qwen2_rm.py](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/configuration_qwen2_rm.py) (6.5 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/generation_config.json) (242.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/merges.txt) (1.6 MB)

- [model-00001-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00001-of-00037.safetensors) (3.6 GB)

- [model-00002-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00002-of-00037.safetensors) (3.7 GB)

- [model-00003-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00003-of-00037.safetensors) (3.6 GB)

- [model-00004-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00004-of-00037.safetensors) (3.7 GB)

- [model-00005-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00005-of-00037.safetensors) (3.7 GB)

- [model-00006-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00006-of-00037.safetensors) (3.7 GB)

- [model-00007-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00007-of-00037.safetensors) (3.6 GB)

- [model-00008-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00008-of-00037.safetensors) (3.7 GB)

- [model-00009-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00009-of-00037.safetensors) (3.7 GB)

- [model-00010-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00010-of-00037.safetensors) (3.7 GB)

- [model-00011-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00011-of-00037.safetensors) (3.6 GB)

- [model-00012-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00012-of-00037.safetensors) (3.7 GB)

- [model-00013-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00013-of-00037.safetensors) (3.7 GB)

- [model-00014-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00014-of-00037.safetensors) (3.7 GB)

- [model-00015-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00015-of-00037.safetensors) (3.6 GB)

- [model-00016-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00016-of-00037.safetensors) (3.7 GB)

- [model-00017-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00017-of-00037.safetensors) (3.7 GB)

- [model-00018-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00018-of-00037.safetensors) (3.7 GB)

- [model-00019-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00019-of-00037.safetensors) (3.6 GB)

- [model-00020-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00020-of-00037.safetensors) (3.7 GB)

- [model-00021-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00021-of-00037.safetensors) (3.7 GB)

- [model-00022-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00022-of-00037.safetensors) (3.7 GB)

- [model-00023-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00023-of-00037.safetensors) (3.6 GB)

- [model-00024-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00024-of-00037.safetensors) (3.7 GB)

- [model-00025-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00025-of-00037.safetensors) (3.7 GB)

- [model-00026-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00026-of-00037.safetensors) (3.7 GB)

- [model-00027-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00027-of-00037.safetensors) (3.6 GB)

- [model-00028-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00028-of-00037.safetensors) (3.7 GB)

- [model-00029-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00029-of-00037.safetensors) (3.7 GB)

- [model-00030-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00030-of-00037.safetensors) (3.7 GB)

- [model-00031-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00031-of-00037.safetensors) (3.6 GB)

- [model-00032-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00032-of-00037.safetensors) (3.7 GB)

- [model-00033-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00033-of-00037.safetensors) (3.7 GB)

- [model-00034-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00034-of-00037.safetensors) (3.7 GB)

- [model-00035-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00035-of-00037.safetensors) (3.6 GB)

- [model-00036-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00036-of-00037.safetensors) (3.7 GB)

- [model-00037-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model-00037-of-00037.safetensors) (3.2 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/model.safetensors.index.json) (77.4 KB)

- [modeling_qwen2_rm.py](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/modeling_qwen2_rm.py) (69.8 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/tokenizer_config.json) (1.3 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-RM-72B/vocab.json) (2.6 MB)


[Back to Main](../../)