
# waifu-diffusion
---


## README([From Huggingface](https://huggingface.co/hakurei/waifu-diffusion))

---
language:
- en
tags:
- stable-diffusion
- text-to-image
license: creativeml-openrail-m
inference: true

---

# waifu-diffusion v1.4 - Diffusion for Weebs

waifu-diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning.

![![image](https://user-images.githubusercontent.com/26317155/210155933-db3a5f1a-1ec3-4777-915c-6deff2841ce9.png)

<sub>masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck</sub>

[Original Weights](https://huggingface.co/hakurei/waifu-diffusion-v1-4)

# Gradio & Colab

We also support a [Gradio](https://github.com/gradio-app/gradio) Web UI and Colab with Diffusers to run Waifu Diffusion:
[![![Open In Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/hakurei/waifu-diffusion-demo)
[![![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_8wPN7dJO746QXsFnB09Uq2VGgSRFuYE#scrollTo=1HaCauSq546O)

## Model Description

[See here for a full model overview.](https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1)

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

## Downstream Uses

This model can be used for entertainment purposes and as a generative art assistant.

## Example Code

```python
import paddle
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    dtype=paddle.float32
).to('cuda')

prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=6)["sample"][0]  
    
image.save("test.png")
```

## Team Members and Acknowledgements

This project would not have been possible without the incredible work by Stability AI and Novel AI.

- [Haru](https://github.com/harubaru)
- [Salt](https://github.com/sALTaccount/)
- [Sta @ Bit192](https://twitter.com/naclbbr)

In order to reach us, you can join our [Discord server](https://discord.gg/touhouai).

[![![Discord Server](https://discordapp.com/api/guilds/930499730843250783/widget.png?style=banner2)](https://discord.gg/touhouai)



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/README.md) (3.3 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/feature_extractor/preprocessor_config.json) (342.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/merges.txt) (512.4 KB)

- [model_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/model_config.json) (661.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/model_index.json) (601.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/model_state.pdparams) (5.1 GB)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/safety_checker/model_config.json) (614.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/scheduler/scheduler_config.json) (342.0 B)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/special_tokens_map.json) (478.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/text_encoder/model_config.json) (487.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/tokenizer/added_tokens.json) (2.0 B)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/tokenizer/merges.txt) (512.4 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/tokenizer/special_tokens_map.json) (478.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/tokenizer/tokenizer_config.json) (291.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/tokenizer/vocab.json) (842.1 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/tokenizer_config.json) (194.0 B)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/unet/config.json) (810.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/vae/config.json) (588.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/vae/model_state.pdparams) (319.1 MB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/vocab.json) (842.1 KB)

- [waifu-diffusion-v1-3.tar.gz](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/waifu-diffusion-v1-3.tar.gz) (4.2 GB)

- [waifu-diffusion.tar.gz](https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/waifu-diffusion.tar.gz) (3.1 GB)


[Back to Main](../../)