
# instruct-pix2pix
---


## README([From Huggingface](https://huggingface.co/timbrooks/instruct-pix2pix))



# InstructPix2Pix: Learning to Follow Image Editing Instructions
GitHub: https://github.com/timothybrooks/instruct-pix2pix
<img src='https://instruct-pix2pix.timothybrooks.com/teaser.jpg'/>



## Example

To use `InstructPix2Pix`, install `diffusers` using `main` for now. The pipeline will be available in the next release

```bash
pip install diffusers accelerate safetensors transformers
```

```python
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)

prompt = "turn him into cyborg"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
images[0]
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/README.md) (1.3 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/feature_extractor/preprocessor_config.json) (518.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/model_index.json) (683.0 B)

- [safety_checker/config.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/safety_checker/config.json) (731.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/scheduler/scheduler_config.json) (600.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/text_encoder/config.json) (691.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/tokenizer/tokenizer_config.json) (842.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/unet/config.json) (1.4 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/vae/config.json) (829.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/timbrooks/instruct-pix2pix/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)