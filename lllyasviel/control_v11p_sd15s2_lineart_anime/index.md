
# control_v11p_sd15s2_lineart_anime
---


## README([From Huggingface](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime))



# Controlnet - v1.1 - *lineart_anime Version*

**Controlnet v1.1** was released in [lllyasviel/ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) by [Lvmin Zhang](https://huggingface.co/lllyasviel).

This checkpoint is a conversion of [the original checkpoint](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15s2_lineart_anime.pth) into `diffusers` format.
It can be used in combination with **Stable Diffusion**, such as [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).


For more details, please also have a look at the [🧨 Diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/controlnet).


ControlNet is a neural network structure to control diffusion models by adding extra conditions. 

![![img](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/./sd.png)

This checkpoint corresponds to the ControlNet conditioned on **lineart_anime images**.

## Model Details
- **Developed by:** Lvmin Zhang, Maneesh Agrawala
- **Model type:** Diffusion-based text-to-image generation model
- **Language(s):** English
- **License:** [The CreativeML OpenRAIL M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license) is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which our license is based.
- **Resources for more information:** [GitHub Repository](https://github.com/lllyasviel/ControlNet), [Paper](https://arxiv.org/abs/2302.05543).
- **Cite as:**

  @misc{zhang2023adding,
    title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
    author={Lvmin Zhang and Maneesh Agrawala},
    year={2023},
    eprint={2302.05543},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }

## Introduction

Controlnet was proposed in [*Adding Conditional Control to Text-to-Image Diffusion Models*](https://arxiv.org/abs/2302.05543) by 
Lvmin Zhang, Maneesh Agrawala.

The abstract reads as follows:

*We present a neural network structure, ControlNet, to control pretrained large diffusion models to support additional input conditions. 
The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small (< 50k). 
Moreover, training a ControlNet is as fast as fine-tuning a diffusion model, and the model can be trained on a personal devices. 
Alternatively, if powerful computation clusters are available, the model can scale to large amounts (millions to billions) of data. 
We report that large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like edge maps, segmentation maps, keypoints, etc. 
This may enrich the methods to control large diffusion models and further facilitate related applications.*

## Example

It is recommended to use the checkpoint with [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as the checkpoint 
has been trained on it.
Experimentally, the checkpoint can be used with other diffusion models such as dreamboothed stable diffusion.

**Note**: If you want to process an image to create the auxiliary conditioning, external dependencies are required as shown below:

1. Install https://github.com/patrickvonplaten/controlnet_aux

```sh
$ pip install controlnet_aux==0.3.0
```

2. Let's install `diffusers` and related packages:

```
$ pip install diffusers transformers accelerate
```

3. Run code:

```python
import paddle
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import LineartAnimeDetector
from paddlenlp.transformers import CLIPTextModel

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

checkpoint = "lllyasviel/control_v11p_sd15s2_lineart_anime"

image = load_image(
    "https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/images/input.png"
)
image = image.resize((512, 512))

prompt = "A warrior girl in the jungle"
processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")

control_image = processor(image)
control_image.save("./images/control.png")

# we skip one layer of the encoder
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder", num_hidden_layers=11, dtype=paddle.float16)

controlnet = ControlNetModel.from_pretrained(checkpoint, dtype=paddle.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", text_encoder=text_encoder, controlnet=controlnet, dtype=paddle.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = paddle.seed(0)
image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

image.save('images/image_out.png')

```

![![bird](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/./images/input.png)

![![bird_canny](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/./images/control.png)

![![bird_canny_out](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/./images/image_out.png)

## Other released checkpoints v1-1

The authors released 14 different checkpoints, each trained with [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
on a different type of conditioning:

| Model Name | Control Image Overview| Condition Image | Control Image Example | Generated Image Example |
|---|---|---|---|---|
|[lllyasviel/control_v11p_sd15_canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)<br/> | *Trained with canny edge detection* | A monochrome image with white edges on a black background.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11e_sd15_ip2p](https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p)<br/> | *Trained with pixel to pixel instruction* | No condition .|<a href="https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11p_sd15_inpaint](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint)<br/> | Trained with image inpainting | No condition.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/images/output.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/images/output.png"/></a>|
|[lllyasviel/control_v11p_sd15_mlsd](https://huggingface.co/lllyasviel/control_v11p_sd15_mlsd)<br/> | Trained with multi-level line segment detection | An image with annotated line segments.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_mlsd/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_mlsd/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_mlsd/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_mlsd/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11f1p_sd15_depth](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth)<br/> | Trained with depth estimation | An image with depth information, usually represented as a grayscale image.|<a href="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11p_sd15_normalbae](https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae)<br/> | Trained with surface normal estimation | An image with surface normal information, usually represented as a color-coded image.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11p_sd15_seg](https://huggingface.co/lllyasviel/control_v11p_sd15_seg)<br/> | Trained with image segmentation | An image with segmented regions, usually represented as a color-coded image.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_seg/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_seg/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_seg/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_seg/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11p_sd15_lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart)<br/> | Trained with line art generation | An image with line art, usually black lines on a white background.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11p_sd15s2_lineart_anime](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime)<br/> | Trained with anime line art generation | An image with anime-style line art.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11p_sd15_openpose](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime)<br/> | Trained with human pose estimation | An image with human poses, usually represented as a set of keypoints or skeletons.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11p_sd15_scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble)<br/> | Trained with scribble-based image generation | An image with scribbles, usually random or user-drawn strokes.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_scribble/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_scribble/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_scribble/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_scribble/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11p_sd15_softedge](https://huggingface.co/lllyasviel/control_v11p_sd15_softedge)<br/> | Trained with soft edge image generation | An image with soft edges, usually to create a more painterly or artistic effect.|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11e_sd15_shuffle](https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle)<br/> | Trained with image shuffling | An image with shuffled patches or regions.|<a href="https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/main/images/control.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/main/images/control.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/main/images/image_out.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/main/images/image_out.png"/></a>|
|[lllyasviel/control_v11f1e_sd15_tile](https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile)<br/> | Trained with image tiling | A blurry image or part of an image .|<a href="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/original.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/original.png"/></a>|<a href="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/output.png"><img width="64" src="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/output.png"/></a>|

## More information

For more information, please also have a look at the [Diffusers ControlNet Blog Post](https://huggingface.co/blog/controlnet) and have a look at the [official docs](https://github.com/lllyasviel/ControlNet-v1-1-nightly).



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/lllyasviel/control_v11p_sd15s2_lineart_anime/README.md) (15.3 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/lllyasviel/control_v11p_sd15s2_lineart_anime/config.json) (1.0 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/lllyasviel/control_v11p_sd15s2_lineart_anime/model_state.pdparams) (1.3 GB)


[Back to Main](../../)