
# JanusFlow-1.3B
---


## README([From Huggingface](https://huggingface.co/deepseek-ai/JanusFlow-1.3B))




## 1. Introduction

We present JanusFlow, a powerful framework that unifies image understanding and generation in a single model. 
JanusFlow introduces a minimalist architecture that integrates autoregressive
language models with rectified flow, a state-of-the-art method in generative modeling. Our
key finding demonstrates that rectified flow can be straightforwardly trained within the large
language model framework, eliminating the need for complex architectural modifications.

[JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2411.07975)

[**Github Repository**](https://github.com/deepseek-ai/Janus)

<div align="center">
<img alt="image" src="teaser.png" style="width:90%;">
</div>


### 2. Model Summary

JanusFlow is a unified understanding and generation MLLM, which decouples visual encoding for multimodal understanding and generation, which is constructed based on DeepSeek-LLM-1.3b-base.
For multimodal understanding, it uses the [SigLIP-L](https://huggingface.co/timm/ViT-L-16-SigLIP-384) as the vision encoder, which supports 384 x 384 image input. 
For image generation, JanusFlow uses rectified flow and [SDXL-VAE](https://huggingface.co/stabilityai/sdxl-vae) to generate 384 x 384 images.
The provided checkpoint is the EMA checkpoint after pre-training and supervised fine-tuning.

<div align="center">
<img alt="image" src="arch.png" style="width:90%;">
</div>


## 3. Quick Start

Please refer to [**Github Repository**](https://github.com/deepseek-ai/Janus)


## 4. License

This code repository is licensed under [the MIT License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-CODE). The use of JanusFlow models is subject to [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-MODEL).


## 5. Citation

```
@misc{ma2024janusflow,
      title={JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation}, 
      author={Yiyang Ma and Xingchao Liu and Xiaokang Chen and Wen Liu and Chengyue Wu and Zhiyu Wu and Zizheng Pan and Zhenda Xie and Haowei Zhang and Xingkai yu and Liang Zhao and Yisong Wang and Jiaying Liu and Chong Ruan},
      journal={arXiv preprint arXiv:2411.07975},
      year={2024}
}
```


## 6. Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](mailto:service@deepseek.com).



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/README.md) (2.6 KB)

- [arch.png](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/arch.png) (257.7 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/config.json) (1.6 KB)

- [image_preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/image_preprocessor_config.json) (452.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/model_state.pdparams) (3.8 GB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/special_tokens_map.json) (525.0 B)

- [teaser.png](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/teaser.png) (3.4 MB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/tokenizer.json) (4.4 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/JanusFlow-1.3B/tokenizer_config.json) (2.9 KB)


[Back to Main](../../)