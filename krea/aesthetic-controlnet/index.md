
# aesthetic-controlnet
---


## README([From Huggingface](https://huggingface.co/krea/aesthetic-controlnet))



# Aesthetic ControlNet
This model can produce highly aesthetic results from an input image and a text prompt.

ControlNet is a method that can be used to condition diffusion models on arbitrary input features, such as image edges, segmentation maps, or human poses. 

Aesthetic ControlNet is a version of this technique that uses image features extracted using a [Canny edge detector](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) and guides a text-to-image diffusion model trained on a large aesthetic dataset.

The base diffusion model is a fine-tuned version of Stable Diffusion 2.1 trained at a resolution of 640x640, and the control network comes from [thibaud/controlnet-sd21](https://huggingface.co/thibaud/controlnet-sd21) by [@thibaudz](https://twitter.com/thibaudz).

For more information about ControlNet, please have a look at this [thread](https://twitter.com/krea_ai/status/1626672218477559809) or at the original [work](https://arxiv.org/pdf/2302.05543.pdf) by Lvmin Zhang and Maneesh Agrawala.

![![Example](https://huggingface.co/krea/aesthetic-controlnet/resolve/main/./examples.jpg)


### Diffusers
Install the following dependencies and then run the code below:

```bash
pip install opencv-python git+https://github.com/huggingface/diffusers.git
```


```py
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

image = load_image("https://huggingface.co/krea/aesthetic-controlnet/resolve/main/krea.jpg")

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

pipe = StableDiffusionControlNetPipeline.from_pretrained("krea/aesthetic-controlnet")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

output = pipe(
    "fantasy flowers",
    canny_image,
    num_inference_steps=20,
    guidance_scale=4,
    width=768,
    height=768,
)

result = output.images[0]
result.save("result.png")
```

## Examples
![![More examples](https://huggingface.co/krea/aesthetic-controlnet/resolve/main/./more_examples.jpg)


## Misuse and Malicious Use
The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

## Authors
Erwann Millon and Victor Perez



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/README.md) (2.6 KB)

- [controlnet/config.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/controlnet/config.json) (1.1 KB)

- [controlnet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/controlnet/model_state.pdparams) (694.7 MB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/feature_extractor/preprocessor_config.json) (520.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/model_index.json) (686.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/scheduler/scheduler_config.json) (377.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/text_encoder/config.json) (778.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/text_encoder/model_state.pdparams) (649.3 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/tokenizer/special_tokens_map.json) (377.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/tokenizer/tokenizer_config.json) (761.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/unet/config.json) (1.4 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/unet/model_state.pdparams) (1.6 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/vae/config.json) (797.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/krea/aesthetic-controlnet/vae/model_state.pdparams) (159.6 MB)


[Back to Main](../../)