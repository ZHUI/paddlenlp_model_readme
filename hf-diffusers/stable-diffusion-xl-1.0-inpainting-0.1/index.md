
# stable-diffusion-xl-1.0-inpainting-0.1
---


## README([From Huggingface](https://huggingface.co/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1))




# SD-XL Inpainting 0.1 Model Card

![![inpaint-example](https://huggingface.co/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/inpaint-examples-min.png)

SD-XL Inpainting 0.1 is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input, with the extra capability of inpainting the pictures by using a mask.

The SD-XL Inpainting 0.1 was initialized with the `stable-diffusion-xl-base-1.0` weights. The model is trained for 40k steps at resolution 1024x1024 and 5% dropping of the text-conditioning to improve classifier-free classifier-free guidance sampling. For inpainting, the UNet has 5 additional input channels (4 for the encoded masked-image and 1 for the mask itself) whose weights were zero-initialized after restoring the non-inpainting checkpoint. During training, we generate synthetic masks and, in 25% mask everything.


## How to use

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", dtype=paddle.float16, variant="fp16").to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a tiger sitting on a park bench"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]
```

**How it works:**
`image`          | `mask_image`
:-------------------------:|:-------------------------:|
<img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png" alt="drawing" width="300"/> | <img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png" alt="drawing" width="300"/>


`prompt`          | `Output`
:-------------------------:|:-------------------------:|
<span style="position: relative;bottom: 150px;">a tiger sitting on a park bench</span> | <img src="https://huggingface.co/datasets/valhalla/images/resolve/main/tiger.png" alt="drawing" width="300"/>

## Model Description

- **Developed by:** The Diffusers team
- **Model type:** Diffusion-based text-to-image generative model
- **License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- **Model Description:** This is a model that can be used to generate and modify images based on text prompts. It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses two fixed, pretrained text encoders ([OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip) and [CLIP-ViT/L](https://github.com/openai/CLIP/tree/main)).


## Uses

### Direct Use

The model is intended for research purposes only. Possible research areas and tasks include

- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.

Excluded uses are described below.

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

## Limitations and Bias

### Limitations

- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.
- The autoencoding part of the model is lossy.
- When the strength parameter is set to 1 (i.e. starting in-painting from a fully masked image), the quality of the image is degraded. The model retains the non-masked contents of the image, but images look less sharp. We're investing this and working on the next version.

### Bias
While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.




## Model Files

- [.gitattributes](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/.gitattributes) (1.5 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/README.md) (4.7 KB)

- [inpaint-examples-min.png](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/inpaint-examples-min.png) (1.6 MB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/model_index.json) (690.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/scheduler/scheduler_config.json) (479.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder/config.json) (746.0 B)

- [text_encoder/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder/model.fp16.safetensors) (234.7 MB)

- [text_encoder/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder/model.safetensors) (469.5 MB)

- [text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder_2/config.json) (758.0 B)

- [text_encoder_2/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder_2/model.fp16.safetensors) (1.3 GB)

- [text_encoder_2/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/text_encoder_2/model.safetensors) (2.6 GB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tokenizer/special_tokens_map.json) (472.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tokenizer/tokenizer_config.json) (737.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tokenizer/vocab.json) (1.0 MB)

- [tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tokenizer_2/merges.txt) (512.3 KB)

- [tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tokenizer_2/special_tokens_map.json) (460.0 B)

- [tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tokenizer_2/tokenizer_config.json) (725.0 B)

- [tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tokenizer_2/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/unet/config.json) (1.9 KB)

- [unet/diffusion_pytorch_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/unet/diffusion_pytorch_model.fp16.safetensors) (4.8 GB)

- [unet/diffusion_pytorch_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/unet/diffusion_pytorch_model.safetensors) (9.6 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/vae/config.json) (659.0 B)

- [vae/diffusion_pytorch_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/vae/diffusion_pytorch_model.fp16.safetensors) (159.6 MB)

- [vae/diffusion_pytorch_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hf-diffusers/stable-diffusion-xl-1.0-inpainting-0.1/vae/diffusion_pytorch_model.safetensors) (319.1 MB)


[Back to Main](../../)