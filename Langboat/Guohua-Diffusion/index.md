
# Guohua-Diffusion
---


## README([From Huggingface](https://huggingface.co/Langboat/Guohua-Diffusion))


# Guohua Diffusion
This is the fine-tuned Stable Diffusion model trained on traditional Chinese paintings.

Use **guohua style** in your prompts for the effect.

## Sample Image
![![example1](https://huggingface.co/Langboat/Guohua-Diffusion/resolve/main/Untitled-1.jpg)
![![example2](https://huggingface.co/Langboat/Guohua-Diffusion/resolve/main/Untitled-3.jpg)

## How to use
#### WebUI
Download the `guohua.ckpt` in model files.
#### Diffusers

This model can be used just like any other Stable Diffusion model. For more information,
please have a look at the [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion).

```python
#!pip install diffusers transformers scipy torch
from diffusers import StableDiffusionPipeline
import torch

model_id = "Langboat/Guohua-Diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=paddle.float16)
pipe = pipe.to("cuda")

prompt = "The Godfather poster in guohua style"
image = pipe(prompt).images[0]

image.save("./the_god_father.png")
```




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/README.md) (1.1 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/feature_extractor/preprocessor_config.json) (407.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/model_index.json) (616.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/safety_checker/model_config.json) (368.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/scheduler/scheduler_config.json) (318.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/text_encoder/model_config.json) (267.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/tokenizer/tokenizer_config.json) (766.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/unet/config.json) (1021.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/vae/config.json) (667.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Langboat/Guohua-Diffusion/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)