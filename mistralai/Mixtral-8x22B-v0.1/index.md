
# Mixtral-8x22B-v0.1
---


## README([From Huggingface](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1))

---
language:
- fr
- it
- de
- es
- en
license: apache-2.0
tags:
- moe
model-index:
- name: Mixtral-8x22B-v0.1
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: AI2 Reasoning Challenge (25-Shot)
      type: ai2_arc
      config: ARC-Challenge
      split: test
      args:
        num_few_shot: 25
    metrics:
    - type: acc_norm
      value: 70.48
      name: normalized accuracy
    source:
      url: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard?query=mistral-community/Mixtral-8x22B-v0.1
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: HellaSwag (10-Shot)
      type: hellaswag
      split: validation
      args:
        num_few_shot: 10
    metrics:
    - type: acc_norm
      value: 88.73
      name: normalized accuracy
    source:
      url: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard?query=mistral-community/Mixtral-8x22B-v0.1
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MMLU (5-Shot)
      type: cais/mmlu
      config: all
      split: test
      args:
        num_few_shot: 5
    metrics:
    - type: acc
      value: 77.81
      name: accuracy
    source:
      url: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard?query=mistral-community/Mixtral-8x22B-v0.1
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: TruthfulQA (0-shot)
      type: truthful_qa
      config: multiple_choice
      split: validation
      args:
        num_few_shot: 0
    metrics:
    - type: mc2
      value: 51.08
    source:
      url: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard?query=mistral-community/Mixtral-8x22B-v0.1
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Winogrande (5-shot)
      type: winogrande
      config: winogrande_xl
      split: validation
      args:
        num_few_shot: 5
    metrics:
    - type: acc
      value: 84.53
      name: accuracy
    source:
      url: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard?query=mistral-community/Mixtral-8x22B-v0.1
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: GSM8k (5-shot)
      type: gsm8k
      config: main
      split: test
      args:
        num_few_shot: 5
    metrics:
    - type: acc
      value: 74.15
      name: accuracy
    source:
      url: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard?query=mistral-community/Mixtral-8x22B-v0.1
      name: Open LLM Leaderboard
---
# Mixtral-8x22B

> [!WARNING]
> This model checkpoint is provided as-is and might not be up-to-date. Please use the corresponding version from https://huggingface.co/mistralai org

> [!TIP]
> MistralAI has uploaded weights to their organization at [mistralai/Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) and [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) too. 

> [!TIP]
> Kudos to [@v2ray](https://huggingface.co/v2ray) for converting the checkpoints and uploading them in `transformers` compatible format. Go give them a follow!

Converted to HuggingFace Transformers format using the script [here](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1/blob/main/convert.py).

The Mixtral-8x22B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts.
## Run the model
```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistral-community/Mixtral-8x22B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pd")

outputs = model.generate(**inputs, max_new_tokens=20)
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

model_id = "mistral-community/Mixtral-8x22B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(0)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pd").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

### Lower precision using (8-bit & 4-bit) using `bitsandbytes`
<details>
<summary> Click to expand </summary>

```diff
+ import torch
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistral-community/Mixtral-8x22B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pd").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

### Load the model with Flash Attention 2
<details>
<summary> Click to expand </summary>

```diff
+ import torch
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistral-community/Mixtral-8x22B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=True)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pd").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

## Notice
Mixtral-8x22B-v0.1 is a pretrained base model and therefore does not have any moderation mechanisms.
# The Mistral AI Team
Albert Jiang, Alexandre Sablayrolles, Alexis Tacnet, Antoine Roux, Arthur Mensch, Audrey Herblin-Stoop, Baptiste Bout, Baudouin de Monicault,Blanche Savary, Bam4d, Caroline Feldman, Devendra Singh Chaplot, Diego de las Casas, Eleonore Arcelin, Emma Bou Hanna, Etienne Metzger, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Harizo Rajaona, Jean-Malo Delignon, Jia Li, Justus Murke, Louis Martin, Louis Ternon, Lucile Saulnier, Lélio Renard Lavaud, Margaret Jennings, Marie Pellat, Marie Torelli, Marie-Anne Lachaux, Nicolas Schuhl, Patrick von Platen, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Thibaut Lavril, Timothée Lacroix, Théophile Gervet, Thomas Wang, Valera Nemychnikova, William El Sayed, William Marshall.
# [Open LLM Leaderboard Evaluation Results](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
Detailed results can be found [here](https://huggingface.co/datasets/open-llm-leaderboard/details_mistral-community__Mixtral-8x22B-v0.1)

|             Metric              |Value|
|---------------------------------|----:|
|Avg.                             |74.46|
|AI2 Reasoning Challenge (25-Shot)|70.48|
|HellaSwag (10-Shot)              |88.73|
|MMLU (5-Shot)                    |77.81|
|TruthfulQA (0-shot)              |51.08|
|Winogrande (5-shot)              |84.53|
|GSM8k (5-shot)                   |74.15|





## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/README.md) (7.4 KB)

- [RELEASE](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/RELEASE) (10.9 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/config.json) (743.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/configuration.json) (73.0 B)

- [convert.py](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/convert.py) (11.5 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/generation_config.json) (116.0 B)

- [model-00001-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00001-of-00059.safetensors) (4.7 GB)

- [model-00002-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00002-of-00059.safetensors) (4.5 GB)

- [model-00003-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00003-of-00059.safetensors) (4.5 GB)

- [model-00004-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00004-of-00059.safetensors) (4.5 GB)

- [model-00005-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00005-of-00059.safetensors) (4.5 GB)

- [model-00006-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00006-of-00059.safetensors) (4.5 GB)

- [model-00007-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00007-of-00059.safetensors) (4.5 GB)

- [model-00008-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00008-of-00059.safetensors) (4.5 GB)

- [model-00009-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00009-of-00059.safetensors) (4.5 GB)

- [model-00010-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00010-of-00059.safetensors) (4.5 GB)

- [model-00011-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00011-of-00059.safetensors) (4.5 GB)

- [model-00012-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00012-of-00059.safetensors) (4.5 GB)

- [model-00013-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00013-of-00059.safetensors) (4.5 GB)

- [model-00014-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00014-of-00059.safetensors) (4.5 GB)

- [model-00015-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00015-of-00059.safetensors) (4.5 GB)

- [model-00016-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00016-of-00059.safetensors) (4.5 GB)

- [model-00017-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00017-of-00059.safetensors) (4.5 GB)

- [model-00018-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00018-of-00059.safetensors) (4.5 GB)

- [model-00019-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00019-of-00059.safetensors) (4.5 GB)

- [model-00020-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00020-of-00059.safetensors) (4.5 GB)

- [model-00021-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00021-of-00059.safetensors) (4.5 GB)

- [model-00022-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00022-of-00059.safetensors) (4.5 GB)

- [model-00023-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00023-of-00059.safetensors) (4.5 GB)

- [model-00024-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00024-of-00059.safetensors) (4.6 GB)

- [model-00025-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00025-of-00059.safetensors) (4.7 GB)

- [model-00026-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00026-of-00059.safetensors) (4.7 GB)

- [model-00027-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00027-of-00059.safetensors) (4.6 GB)

- [model-00028-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00028-of-00059.safetensors) (4.5 GB)

- [model-00029-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00029-of-00059.safetensors) (4.5 GB)

- [model-00030-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00030-of-00059.safetensors) (4.5 GB)

- [model-00031-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00031-of-00059.safetensors) (4.5 GB)

- [model-00032-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00032-of-00059.safetensors) (4.5 GB)

- [model-00033-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00033-of-00059.safetensors) (4.5 GB)

- [model-00034-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00034-of-00059.safetensors) (4.5 GB)

- [model-00035-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00035-of-00059.safetensors) (4.5 GB)

- [model-00036-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00036-of-00059.safetensors) (4.5 GB)

- [model-00037-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00037-of-00059.safetensors) (4.5 GB)

- [model-00038-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00038-of-00059.safetensors) (4.5 GB)

- [model-00039-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00039-of-00059.safetensors) (4.5 GB)

- [model-00040-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00040-of-00059.safetensors) (4.5 GB)

- [model-00041-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00041-of-00059.safetensors) (4.5 GB)

- [model-00042-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00042-of-00059.safetensors) (4.5 GB)

- [model-00043-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00043-of-00059.safetensors) (4.5 GB)

- [model-00044-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00044-of-00059.safetensors) (4.5 GB)

- [model-00045-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00045-of-00059.safetensors) (4.5 GB)

- [model-00046-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00046-of-00059.safetensors) (4.5 GB)

- [model-00047-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00047-of-00059.safetensors) (4.5 GB)

- [model-00048-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00048-of-00059.safetensors) (4.5 GB)

- [model-00049-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00049-of-00059.safetensors) (4.5 GB)

- [model-00050-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00050-of-00059.safetensors) (4.5 GB)

- [model-00051-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00051-of-00059.safetensors) (4.5 GB)

- [model-00052-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00052-of-00059.safetensors) (4.6 GB)

- [model-00053-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00053-of-00059.safetensors) (4.7 GB)

- [model-00054-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00054-of-00059.safetensors) (4.7 GB)

- [model-00055-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00055-of-00059.safetensors) (4.6 GB)

- [model-00056-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00056-of-00059.safetensors) (4.5 GB)

- [model-00057-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00057-of-00059.safetensors) (4.5 GB)

- [model-00058-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00058-of-00059.safetensors) (4.5 GB)

- [model-00059-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model-00059-of-00059.safetensors) (951.0 MB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/model.safetensors.index.json) (161.8 KB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/sentencepiece.bpe.model) (481.9 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/special_tokens_map.json) (72.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/tokenizer.json) (1.7 MB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/tokenizer.model) (481.9 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-v0.1/tokenizer_config.json) (967.0 B)


[Back to Main](../../)