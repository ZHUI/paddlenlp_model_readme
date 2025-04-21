
# Janus-Pro-1B
---


## README([From Huggingface](https://huggingface.co/deepseek-ai/Janus-Pro-1B))



## 1. Introduction

Janus-Pro is a novel autoregressive framework that unifies multimodal understanding and generation. 
It addresses the limitations of previous approaches by decoupling visual encoding into separate pathways, while still utilizing a single, unified transformer architecture for processing. The decoupling not only alleviates the conflict between the visual encoder’s roles in understanding and generation, but also enhances the framework’s flexibility. 
Janus-Pro surpasses previous unified model and matches or exceeds the performance of task-specific models. 
The simplicity, high flexibility, and effectiveness of Janus-Pro make it a strong candidate for next-generation unified multimodal models.

[**Github Repository**](https://github.com/deepseek-ai/Janus)

<div align="center">
<img alt="image" src="janus_pro_teaser1.png" style="width:90%;">
</div>

<div align="center">
<img alt="image" src="janus_pro_teaser2.png" style="width:90%;">
</div>


### 2. Model Summary

Janus-Pro is a unified understanding and generation MLLM, which decouples visual encoding for multimodal understanding and generation. 
Janus-Pro is constructed based on the DeepSeek-LLM-1.5b-base/DeepSeek-LLM-7b-base.

For multimodal understanding, it uses the [SigLIP-L](https://huggingface.co/timm/ViT-L-16-SigLIP-384) as the vision encoder, which supports 384 x 384 image input. For image generation, Janus-Pro uses the tokenizer from [here](https://github.com/FoundationVision/LlamaGen) with a downsample rate of 16.



## 3. Quick Start

Please refer to [**Github Repository**](https://github.com/deepseek-ai/Janus)


## 4. License

This code repository is licensed under [the MIT License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-CODE). The use of Janus-Pro models is subject to [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-MODEL).
## 5. Citation

```
@article{chen2025janus,
  title={Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling},
  author={Chen, Xiaokang and Wu, Zhiyu and Liu, Xingchao and Pan, Zizheng and Liu, Wen and Xie, Zhenda and Yu, Xingkai and Ruan, Chong},
  journal={arXiv preprint arXiv:2501.17811},
  year={2025}
}
```

## 6. Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](mailto:service@deepseek.com).



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/Janus-Pro-1B/README.md) (2.5 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/Janus-Pro-1B/config.json) (1.4 KB)

- [image_preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/Janus-Pro-1B/image_preprocessor_config.json) (346.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/Janus-Pro-1B/model_state.pdparams) (3.9 GB)

- [processor_config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/Janus-Pro-1B/processor_config.json) (210.0 B)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/Janus-Pro-1B/special_tokens_map.json) (344.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/Janus-Pro-1B/tokenizer.json) (4.5 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/Janus-Pro-1B/tokenizer_config.json) (285.0 B)


[Back to Main](../../)