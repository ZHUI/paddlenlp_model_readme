
# qwen1.5-moe-a2.7b
---


## README([From Huggingface](https://huggingface.co/qwen/qwen1.5-moe-a2.7b))



# Qwen1.5-MoE-A2.7B


## Introduction

Qwen1.5-MoE is a transformer-based MoE decoder-only language model pretrained on a large amount of data. 

For more details, please refer to our [blog post](https://qwenlm.github.io/blog/qwen-moe/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).

## Model Details
Qwen1.5-MoE employs Mixture of Experts (MoE) architecture, where the models are upcycled from dense language models. For instance, `Qwen1.5-MoE-A2.7B` is upcycled from `Qwen-1.8B`. It has 14.3B parameters in total and 2.7B activated parameters during runtime, while achieving comparable performance to `Qwen1.5-7B`, it only requires 25% of the training resources. We also observed that the inference speed is 1.74 times that of `Qwen1.5-7B`.

## Requirements
The code of Qwen1.5-MoE has been in the latest Hugging face transformers and we advise you to build from source with command `pip install git+https://github.com/huggingface/transformers`, or you might encounter the following error:
```
KeyError: 'qwen2_moe'.
```

## Usage

We do not advise you to use base language models for text generation. Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., on this model.




## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/LICENSE) (6.7 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/README.md) (1.4 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/config.json) (1.1 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/generation_config.json) (104.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/merges.txt) (1.6 MB)

- [model-00001-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/model-00001-of-00007.safetensors) (3.8 GB)

- [model-00002-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/model-00002-of-00007.safetensors) (4.3 GB)

- [model-00003-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/model-00003-of-00007.safetensors) (4.3 GB)

- [model-00004-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/model-00004-of-00007.safetensors) (4.3 GB)

- [model-00005-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/model-00005-of-00007.safetensors) (4.3 GB)

- [model-00006-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/model-00006-of-00007.safetensors) (4.3 GB)

- [model-00007-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/model-00007-of-00007.safetensors) (1.6 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/model.safetensors.index.json) (420.3 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/tokenizer_config.json) (1.3 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/qwen/qwen1.5-moe-a2.7b/vocab.json) (2.6 MB)


[Back to Main](../../)