
# Qwen2.5-Math-72B
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2.5-Math-72B))




# Qwen2.5-Math-72B

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
> **Qwen2.5-Math-72B-Instruct** is an instruction model for chatting;
>
> **Qwen2.5-Math-72B** is a base model typically used for completion and few-shot inference, serving as a better starting point for fine-tuning.
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

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/LICENSE) (6.8 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/README.md) (3.2 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/config.json) (672.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/configuration.json) (2.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/generation_config.json) (138.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/merges.txt) (1.6 MB)

- [model-00001-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00001-of-00037.safetensors) (3.5 GB)

- [model-00002-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00002-of-00037.safetensors) (3.7 GB)

- [model-00003-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00003-of-00037.safetensors) (3.6 GB)

- [model-00004-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00004-of-00037.safetensors) (3.7 GB)

- [model-00005-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00005-of-00037.safetensors) (3.7 GB)

- [model-00006-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00006-of-00037.safetensors) (3.7 GB)

- [model-00007-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00007-of-00037.safetensors) (3.6 GB)

- [model-00008-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00008-of-00037.safetensors) (3.7 GB)

- [model-00009-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00009-of-00037.safetensors) (3.7 GB)

- [model-00010-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00010-of-00037.safetensors) (3.7 GB)

- [model-00011-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00011-of-00037.safetensors) (3.6 GB)

- [model-00012-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00012-of-00037.safetensors) (3.7 GB)

- [model-00013-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00013-of-00037.safetensors) (3.7 GB)

- [model-00014-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00014-of-00037.safetensors) (3.7 GB)

- [model-00015-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00015-of-00037.safetensors) (3.6 GB)

- [model-00016-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00016-of-00037.safetensors) (3.7 GB)

- [model-00017-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00017-of-00037.safetensors) (3.7 GB)

- [model-00018-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00018-of-00037.safetensors) (3.7 GB)

- [model-00019-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00019-of-00037.safetensors) (3.6 GB)

- [model-00020-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00020-of-00037.safetensors) (3.7 GB)

- [model-00021-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00021-of-00037.safetensors) (3.7 GB)

- [model-00022-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00022-of-00037.safetensors) (3.7 GB)

- [model-00023-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00023-of-00037.safetensors) (3.6 GB)

- [model-00024-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00024-of-00037.safetensors) (3.7 GB)

- [model-00025-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00025-of-00037.safetensors) (3.7 GB)

- [model-00026-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00026-of-00037.safetensors) (3.7 GB)

- [model-00027-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00027-of-00037.safetensors) (3.6 GB)

- [model-00028-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00028-of-00037.safetensors) (3.7 GB)

- [model-00029-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00029-of-00037.safetensors) (3.7 GB)

- [model-00030-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00030-of-00037.safetensors) (3.7 GB)

- [model-00031-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00031-of-00037.safetensors) (3.6 GB)

- [model-00032-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00032-of-00037.safetensors) (3.7 GB)

- [model-00033-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00033-of-00037.safetensors) (3.7 GB)

- [model-00034-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00034-of-00037.safetensors) (3.7 GB)

- [model-00035-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00035-of-00037.safetensors) (3.6 GB)

- [model-00036-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00036-of-00037.safetensors) (3.7 GB)

- [model-00037-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model-00037-of-00037.safetensors) (3.2 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/model.safetensors.index.json) (77.2 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/tokenizer_config.json) (7.1 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Math-72B/vocab.json) (2.6 MB)


[Back to Main](../../)