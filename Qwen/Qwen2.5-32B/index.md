
# Qwen2.5-32B
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2.5-32B))



# Qwen2.5-32B

## Introduction

Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters. Qwen2.5 brings the following improvements upon Qwen2:

- Significantly **more knowledge** and has greatly improved capabilities in **coding** and **mathematics**, thanks to our specialized expert models in these domains.
- Significant improvements in **instruction following**, **generating long texts** (over 8K tokens), **understanding structured data** (e.g, tables), and **generating structured outputs** especially JSON. **More resilient to the diversity of system prompts**, enhancing role-play implementation and condition-setting for chatbots.
- **Long-context Support** up to 128K tokens and can generate up to 8K tokens.
- **Multilingual support** for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more. 

**This repo contains the base 32B Qwen2.5 model**, which has the following features:
- Type: Causal Language Models
- Training Stage: Pretraining
- Architecture: transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias
- Number of Parameters: 32.5B
- Number of Paramaters (Non-Embedding): 31.0B
- Number of Layers: 64
- Number of Attention Heads (GQA): 40 for Q and 8 for KV
- Context Length: 131,072 tokens

**We do not recommend using base language models for conversations.** Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., on this model.

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2.5/), [GitHub](https://github.com/QwenLM/Qwen2.5), and [Documentation](https://qwen.readthedocs.io/en/latest/).

## Requirements

The code of Qwen2.5 has been in the latest Hugging face `transformers` and we advise you to use the latest version of `transformers`.

With `transformers<4.37.0`, you will encounter the following error:
```
KeyError: 'qwen2'
```

## Evaluation & Performance

Detailed evaluation results are reported in this [📑 blog](https://qwenlm.github.io/blog/qwen2.5/).

For requirements on GPU memory and the respective throughput, see results [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{qwen2.5,
    title = {Qwen2.5: A Party of Foundation Models},
    url = {https://qwenlm.github.io/blog/qwen2.5/},
    author = {Qwen Team},
    month = {September},
    year = {2024}
}

@article{qwen2,
      title={Qwen2 Technical Report}, 
      author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
      journal={arXiv preprint arXiv:2407.10671},
      year={2024}
}
```



## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/LICENSE) (11.1 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/README.md) (3.7 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/config.json) (658.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/configuration.json) (2.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/generation_config.json) (138.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/merges.txt) (1.6 MB)

- [model-00001-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00001-of-00017.safetensors) (3.6 GB)

- [model-00002-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00002-of-00017.safetensors) (3.6 GB)

- [model-00003-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00003-of-00017.safetensors) (3.6 GB)

- [model-00004-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00004-of-00017.safetensors) (3.6 GB)

- [model-00005-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00005-of-00017.safetensors) (3.6 GB)

- [model-00006-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00006-of-00017.safetensors) (3.6 GB)

- [model-00007-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00007-of-00017.safetensors) (3.6 GB)

- [model-00008-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00008-of-00017.safetensors) (3.6 GB)

- [model-00009-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00009-of-00017.safetensors) (3.6 GB)

- [model-00010-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00010-of-00017.safetensors) (3.6 GB)

- [model-00011-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00011-of-00017.safetensors) (3.6 GB)

- [model-00012-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00012-of-00017.safetensors) (3.6 GB)

- [model-00013-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00013-of-00017.safetensors) (3.6 GB)

- [model-00014-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00014-of-00017.safetensors) (3.6 GB)

- [model-00015-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00015-of-00017.safetensors) (3.6 GB)

- [model-00016-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00016-of-00017.safetensors) (3.6 GB)

- [model-00017-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model-00017-of-00017.safetensors) (2.9 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/model.safetensors.index.json) (61.8 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/tokenizer_config.json) (7.1 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-32B/vocab.json) (2.6 MB)


[Back to Main](../../)