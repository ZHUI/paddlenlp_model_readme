
# waifu-diffusion-v1-3
---


## README([From Huggingface](https://huggingface.co/hakurei/waifu-diffusion-v1-3))

---
language:
- en
tags:
- stable-diffusion
- text-to-image
license: creativeml-openrail-m
inference: false

---

# Waifu Diffusion v1.3

Waifu Diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning.

- [Float 16 EMA Pruned](https://huggingface.co/hakurei/waifu-diffusion-v1-3/blob/main/wd-v1-3-float16.ckpt)
- [Float 32 EMA Pruned](https://huggingface.co/hakurei/waifu-diffusion-v1-3/blob/main/wd-v1-3-float32.ckpt)
- [Float 32 Full Weights](https://huggingface.co/hakurei/waifu-diffusion-v1-3/blob/main/wd-v1-3-full.ckpt)
- [Float 32 Full Weights + Optimizer Weights (For Training)](https://huggingface.co/hakurei/waifu-diffusion-v1-3/blob/main/wd-v1-3-full-opt.ckpt)

## Model Description

The model originally used for fine-tuning is [Stable Diffusion 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4), which is a latent image diffusion model trained on [LAION2B-en](https://huggingface.co/datasets/laion/laion2B-en). The current model has been fine-tuned with a learning rate of 5.0e-6 for 10 epochs on 680k anime-styled images.

[See here for an in-depth overview of Waifu Diffusion 1.3.](https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1)

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

## Downstream Uses

This model can be used for entertainment purposes and as a generative art assistant.

## Team Members and Acknowledgements

This project would not have been possible without the incredible work by the [CompVis Researchers](https://ommer-lab.com/).

- [Anthony Mercurio](https://github.com/harubaru)
- [Salt](https://github.com/sALTaccount/)
- [Cafe](https://twitter.com/cafeai_labs)

In order to reach us, you can join our [Discord server](https://discord.gg/touhouai).

[![![Discord Server](https://discordapp.com/api/guilds/930499730843250783/widget.png?style=banner2)](https://discord.gg/touhouai)



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/README.md) (2.7 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/feature_extractor/preprocessor_config.json) (342.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/model_index.json) (601.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/safety_checker/model_config.json) (614.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/scheduler/scheduler_config.json) (342.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/text_encoder/model_config.json) (487.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/tokenizer/merges.txt) (512.4 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/tokenizer/special_tokens_map.json) (478.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/tokenizer/tokenizer_config.json) (312.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/tokenizer/vocab.json) (842.1 KB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/unet/config.json) (807.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/vae/config.json) (610.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion-v1-3/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)