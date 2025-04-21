
# Mistral-7B-v0.1
---


## README([From Huggingface](https://huggingface.co/mistralai/Mistral-7B-v0.1))

---
language:
- en
license: apache-2.0
tags:
- pretrained
pipeline_tag: text-generation
inference:
  parameters:
    temperature: 0.7

extra_gated_description: If you want to learn more about how we process your personal data, please read our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
---

# Model Card for Mistral-7B-v0.1

The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. 
Mistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks we tested.

For full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/announcing-mistral-7b/).

## Model Architecture

Mistral-7B-v0.1 is a transformer model, with the following architecture choices:
- Grouped-Query Attention
- Sliding-Window Attention
- Byte-fallback BPE tokenizer

## Troubleshooting

- If you see the following error:
```
KeyError: 'mistral'
```
- Or:
```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

Ensure you are utilizing a stable version of Transformers, 4.34.0 or newer.

## Notice

Mistral 7B is a pretrained base model and therefore does not have any moderation mechanisms.

## The Mistral AI Team
 
Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.



## Model Files

- [.mdl](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/.mdl) (52.0 B)

- [.msc](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/.msc) (849.0 B)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/README.md) (1.5 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/config.json) (532.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/generation_config.json) (45.0 B)

- [model-00001-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/model-00001-of-00003.safetensors) (4.6 GB)

- [model-00002-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/model-00002-of-00003.safetensors) (4.7 GB)

- [model-00003-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/model-00003-of-00003.safetensors) (4.2 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/model.safetensors.index.json) (24.0 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/model_state.pdparams) (13.5 GB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/sentencepiece.bpe.model) (481.9 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/special_tokens_map.json) (72.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/tokenizer.json) (1.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mistral-7B-v0.1/tokenizer_config.json) (966.0 B)


[Back to Main](../../)