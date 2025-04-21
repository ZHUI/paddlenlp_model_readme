
# Qwen2-VL-7B
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2-VL-7B))



# Qwen2-VL-7B

## Introduction

We're excited to unveil **Qwen2-VL**, the latest iteration of our Qwen-VL model, representing nearly a year of innovation. 

> [!Important]
> This is the base pretrained model of Qwen2-VL-7B without instruction tuning.

### What’s New in Qwen2-VL?

#### Key Enhancements:

* **SoTA understanding of images of various resolution & ratio**: Qwen2-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.
* **Understanding videos of 20min+**: Qwen2-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.
* **Agent that can operate your mobiles, robots, etc.**: with the abilities of complex reasoning and decision making, Qwen2-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.
* **Multilingual Support**: to serve global users, besides English and Chinese, Qwen2-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.

#### Model Architecture Updates:

* **Naive Dynamic Resolution**: Unlike before, Qwen2-VL can handle arbitrary image resolutions, mapping them into a dynamic number of visual tokens, offering a more human-like visual processing experience.

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/qwen2_vl.jpg" width="80%"/>
<p>

* **Multimodal Rotary Position Embedding (M-ROPE)**: Decomposes positional embedding into parts to capture 1D textual, 2D visual, and 3D video positional information, enhancing its multimodal processing capabilities.

<p align="center">
    <img src="http://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/mrope.png" width="80%"/>
<p>

We have three models with 2, 7 and 72 billion parameters. 

This repo contains the **pretrained** 7B Qwen2-VL model. 

For more information, visit our [Blog](https://qwenlm.github.io/blog/qwen2-vl/) and [GitHub](https://github.com/QwenLM/Qwen2-VL).

## Requirements

The code of Qwen2-VL has been in the latest Hugging Face `transformers` and we advise you to install the latest version with command `pip install -U transformers`, or you might encounter the following error:

```
KeyError: 'qwen2_vl'
```


## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{Qwen2-VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution}, 
  author={Peng Wang and Shuai Bai and Sinan Tan and Shijie Wang and Zhihao Fan and Jinze Bai and Keqin Chen and Xuejing Liu and Jialin Wang and Wenbin Ge and Yang Fan and Kai Dang and Mengfei Du and Xuancheng Ren and Rui Men and Dayiheng Liu and Chang Zhou and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```




## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/LICENSE) (11.1 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/README.md) (3.4 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/config.json) (1.1 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/generation_config.json) (209.0 B)

- [image_preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/image_preprocessor_config.json) (347.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/merges.txt) (1.6 MB)

- [model-00001-of-00005.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/model-00001-of-00005.safetensors) (3.6 GB)

- [model-00002-of-00005.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/model-00002-of-00005.safetensors) (3.6 GB)

- [model-00003-of-00005.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/model-00003-of-00005.safetensors) (3.6 GB)

- [model-00004-of-00005.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/model-00004-of-00005.safetensors) (3.6 GB)

- [model-00005-of-00005.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/model-00005-of-00005.safetensors) (1.0 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/model.safetensors.index.json) (55.1 KB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/preprocessor_config.json) (347.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/tokenizer_config.json) (4.1 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-VL-7B/vocab.json) (2.6 MB)


[Back to Main](../../)