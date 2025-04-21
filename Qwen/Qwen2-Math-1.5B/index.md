
# Qwen2-Math-1.5B
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2-Math-1.5B))




# Qwen2-Math-1.5B

> [!Warning]
> <div align="center">
> <b>
> 🚨 Temporarily this model mainly supports English. We will release bilingual (English & Chinese) models soon!
> </b>
> </div>

## Introduction

Over the past year, we have dedicated significant effort to researching and enhancing the reasoning capabilities of large language models, with a particular focus on their ability to solve arithmetic and mathematical problems. Today, we are delighted to introduce a serise of math-specific large language models of our Qwen2 series,  Qwen2-Math and Qwen2-Math-Instruct-1.5B/7B/72B. Qwen2-Math is a series of specialized math language models built upon the Qwen2 LLMs, which significantly outperforms the mathematical capabilities of open-source models and even closed-source models (e.g., GPT4o). We hope that Qwen2-Math can contribute to the scientific community for solving advanced mathematical problems that require complex, multi-step logical reasoning.


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

> [!Important]
>
> **Qwen2-Math-1.5B-Instruct** is an instruction model for chatting;
>
> **Qwen2-Math-1.5B** is a base model typically used for completion and few-shot inference, serving as a better starting point for fine-tuning.
> 

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

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/README.md) (2.2 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/config.json) (671.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/configuration.json) (73.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/generation_config.json) (138.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/merges.txt) (1.6 MB)

- [model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/model.safetensors) (2.9 GB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/tokenizer_config.json) (1.3 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-Math-1.5B/vocab.json) (2.6 MB)


[Back to Main](../../)