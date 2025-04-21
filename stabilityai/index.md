
# stabilityai
---


## README([From Huggingface](https://huggingface.co/stabilityai))


# Stable Diffusion 3 Medium
![![sd3 demo images](https://huggingface.co/stabilityai/resolve/main/sd3demo.jpg)

## Model

![![mmdit](https://huggingface.co/stabilityai/resolve/main/mmdit.png)

[Stable Diffusion 3 Medium](stability.ai/news/stable-diffusion-3-medium) is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that features greatly improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.

For more technical details, please refer to the [Research paper](https://stability.ai/news/stable-diffusion-3-research-paper).

Please note: this model is released under the Stability Non-Commercial Research Community License. For a Creator License or an Enterprise License visit Stability.ai or [contact us](https://stability.ai/license) for commercial licensing details.

### Model Description

- **Developed by:** Stability AI
- **Model type:** MMDiT text-to-image generative model
- **Model Description:** This is a model that can be used to generate images based on text prompts. It is a Multimodal Diffusion Transformer
(https://arxiv.org/abs/2403.03206) that uses three fixed, pretrained text encoders 
([OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip), [CLIP-ViT/L](https://github.com/openai/CLIP/tree/main) and [T5-xxl](https://huggingface.co/google/t5-v1_1-xxl))

### License

- **Non-commercial Use:** Stable Diffusion 3 Medium is released under the [Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE). The model is free to use for non-commercial purposes such as academic research.
- **Commercial Use**: This model is not available for commercial use without a separate commercial license from Stability. We encourage professional artists, designers, and creators to use our Creator License. Please visit https://stability.ai/license to learn more.


### Model Sources

For local or self-hosted use, we recommend [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for inference.

Stable Diffusion 3 Medium is available on our [Stability API Platform](https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post). 

Stable Diffusion 3 models and workflows are available on [Stable Assistant](https://stability.ai/stable-assistant) and on Discord via [Stable Artisan](https://stability.ai/stable-artisan). 

- **ComfyUI:** https://github.com/comfyanonymous/ComfyUI
- **StableSwarmUI:** https://github.com/Stability-AI/StableSwarmUI
- **Tech report:** https://stability.ai/news/stable-diffusion-3-research-paper
- **Demo:** https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium


## Training Dataset

We used synthetic data and filtered publicly available data to train our models. The model was pre-trained on 1 billion images. The fine-tuning data includes 30M high-quality aesthetic images focused on specific visual content and style, as well as 3M preference data images.

## Using with Diffusers

Make sure you upgrade to the latest version of `diffusers`: `pip install -U diffusers`. And then you can run:

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", dtype=paddle.float16)
pipe = pipe

image = pipe(
	"A cat holding a sign that says hello world",
	negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image
```

Refer to [the documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3) for more details on optimization and image-to-image support. 

## Uses

### Intended Uses

Intended uses include the following: 
* Generation of artworks and use in design and other artistic processes.
* Applications in educational or creative tools.
* Research on generative models, including understanding the limitations of generative models.

All uses of the model should be in accordance with our [Acceptable Use Policy](https://stability.ai/use-policy).

### Out-of-Scope Uses

The model was not trained to be factual or true representations of people or events.  As such, using the model to generate such content is out-of-scope of the abilities of this model.

## Safety

As part of our safety-by-design and responsible AI deployment approach, we implement safety measures throughout the development of our models, from the time we begin pre-training a model to the ongoing development, fine-tuning, and deployment of each model. We have implemented a number of safety mitigations that are intended to reduce the risk of severe harms, however we recommend that developers conduct their own testing and apply additional mitigations based on their specific use cases.  
For more about our approach to Safety, please visit our [Safety page](https://stability.ai/safety).

### Evaluation Approach

Our evaluation methods include structured evaluations and internal and external red-teaming testing for specific, severe harms such as child sexual abuse and exploitation, extreme violence, and gore, sexually explicit content, and non-consensual nudity.  Testing was conducted primarily in English and may not cover all possible harms.  As with any model, the model may, at times, produce inaccurate, biased or objectionable responses to user prompts. 

### Risks identified and mitigations:

* Harmful content:  We have used filtered data sets when training our models and implemented safeguards that attempt to strike the right balance between usefulness and preventing harm. However, this does not guarantee that all possible harmful content has been removed. The model may, at times, generate toxic or biased content.  All developers and deployers should exercise caution and implement content safety guardrails based on their specific product policies and application use cases.
* Misuse: Technical limitations and developer and end-user education can help mitigate against malicious applications of models. All users are required to adhere to our Acceptable Use Policy, including when applying fine-tuning and prompt engineering mechanisms. Please reference the Stability AI Acceptable Use Policy for information on violative uses of our products.
* Privacy violations: Developers and deployers are encouraged to adhere to privacy regulations with techniques that respect data privacy.

### Contact

Please report any issues with the model or contact us:

* Safety issues:  safety@stability.ai
* Security issues:  security@stability.ai
* Privacy issues:  privacy@stability.ai 
* License and general: https://stability.ai/license
* Enterprise license: https://stability.ai/enterprise




## Model Files

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/model_index.json) (856.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/scheduler/scheduler_config.json) (138.0 B)

- [sd-vae-ft-ema/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-vae-ft-ema/config.json) (729.0 B)

- [sd-vae-ft-ema/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-vae-ft-ema/model_state.pdparams) (319.1 MB)

- [sd-vae-ft-mse/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-vae-ft-mse/config.json) (729.0 B)

- [sd-vae-ft-mse/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-vae-ft-mse/model_state.pdparams) (319.1 MB)

- [sd-x2-latent-upscaler/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/model_index.json) (465.0 B)

- [sd-x2-latent-upscaler/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/scheduler/scheduler_config.json) (332.0 B)

- [sd-x2-latent-upscaler/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/text_encoder/config.json) (719.0 B)

- [sd-x2-latent-upscaler/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/text_encoder/model_state.pdparams) (469.5 MB)

- [sd-x2-latent-upscaler/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/tokenizer/merges.txt) (512.3 KB)

- [sd-x2-latent-upscaler/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/tokenizer/special_tokens_map.json) (389.0 B)

- [sd-x2-latent-upscaler/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/tokenizer/tokenizer_config.json) (874.0 B)

- [sd-x2-latent-upscaler/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/tokenizer/vocab.json) (1.0 MB)

- [sd-x2-latent-upscaler/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/unet/config.json) (1.4 KB)

- [sd-x2-latent-upscaler/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/unet/model_state.pdparams) (1.4 GB)

- [sd-x2-latent-upscaler/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/vae/config.json) (836.0 B)

- [sd-x2-latent-upscaler/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sd-x2-latent-upscaler/vae/model_state.pdparams) (319.1 MB)

- [sdxl-turbo/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/model_index.json) (634.0 B)

- [sdxl-turbo/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/scheduler/scheduler_config.json) (456.0 B)

- [sdxl-turbo/sd_xl_turbo_1.0.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/sd_xl_turbo_1.0.safetensors) (12.9 GB)

- [sdxl-turbo/sd_xl_turbo_1.0_fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/sd_xl_turbo_1.0_fp16.safetensors) (6.5 GB)

- [sdxl-turbo/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/text_encoder/config.json) (584.0 B)

- [sdxl-turbo/text_encoder/model_state.fp16.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/text_encoder/model_state.fp16.pdparams) (234.7 MB)

- [sdxl-turbo/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/text_encoder/model_state.pdparams) (469.5 MB)

- [sdxl-turbo/text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/text_encoder_2/config.json) (594.0 B)

- [sdxl-turbo/text_encoder_2/model_state.fp16.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/text_encoder_2/model_state.fp16.pdparams) (1.3 GB)

- [sdxl-turbo/text_encoder_2/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/text_encoder_2/model_state.pdparams) (2.6 GB)

- [sdxl-turbo/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/tokenizer/merges.txt) (512.3 KB)

- [sdxl-turbo/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/tokenizer/special_tokens_map.json) (479.0 B)

- [sdxl-turbo/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/tokenizer/tokenizer_config.json) (667.0 B)

- [sdxl-turbo/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/tokenizer/vocab.json) (1.0 MB)

- [sdxl-turbo/tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/tokenizer_2/merges.txt) (512.3 KB)

- [sdxl-turbo/tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/tokenizer_2/special_tokens_map.json) (377.0 B)

- [sdxl-turbo/tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/tokenizer_2/tokenizer_config.json) (774.0 B)

- [sdxl-turbo/tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/tokenizer_2/vocab.json) (1.0 MB)

- [sdxl-turbo/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/unet/config.json) (1.8 KB)

- [sdxl-turbo/unet/model_state.fp16.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/unet/model_state.fp16.pdparams) (4.8 GB)

- [sdxl-turbo/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/unet/model_state.pdparams) (9.6 GB)

- [sdxl-turbo/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/vae/config.json) (672.0 B)

- [sdxl-turbo/vae/model_state.fp16.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/vae/model_state.fp16.pdparams) (159.6 MB)

- [sdxl-turbo/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-turbo/vae/model_state.pdparams) (319.1 MB)

- [sdxl-vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-vae/config.json) (647.0 B)

- [sdxl-vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-vae/model_state.pdparams) (319.1 MB)

- [sdxl-vae/sdxl_vae.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/sdxl-vae/sdxl_vae.safetensors) (319.1 MB)

- [stable-diffusion-2-1-base/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/model_index.json) (539.0 B)

- [stable-diffusion-2-1-base/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/scheduler/scheduler_config.json) (342.0 B)

- [stable-diffusion-2-1-base/text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/text_encoder/model_config.json) (262.0 B)

- [stable-diffusion-2-1-base/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/text_encoder/model_state.pdparams) (1.3 GB)

- [stable-diffusion-2-1-base/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/tokenizer/merges.txt) (512.4 KB)

- [stable-diffusion-2-1-base/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/tokenizer/special_tokens_map.json) (466.0 B)

- [stable-diffusion-2-1-base/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/tokenizer/tokenizer_config.json) (781.0 B)

- [stable-diffusion-2-1-base/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/tokenizer/vocab.json) (842.1 KB)

- [stable-diffusion-2-1-base/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/unet/config.json) (907.0 B)

- [stable-diffusion-2-1-base/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/unet/model_state.pdparams) (3.2 GB)

- [stable-diffusion-2-1-base/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/vae/config.json) (617.0 B)

- [stable-diffusion-2-1-base/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-base/vae/model_state.pdparams) (319.1 MB)

- [stable-diffusion-2-1-unclip-small/feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/feature_extractor/preprocessor_config.json) (466.0 B)

- [stable-diffusion-2-1-unclip-small/image_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/image_encoder/config.json) (639.0 B)

- [stable-diffusion-2-1-unclip-small/image_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/image_encoder/model_state.pdparams) (1.1 GB)

- [stable-diffusion-2-1-unclip-small/image_noising_scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/image_noising_scheduler/scheduler_config.json) (455.0 B)

- [stable-diffusion-2-1-unclip-small/image_normalizer/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/image_normalizer/config.json) (319.0 B)

- [stable-diffusion-2-1-unclip-small/image_normalizer/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/image_normalizer/model_state.pdparams) (6.3 KB)

- [stable-diffusion-2-1-unclip-small/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/model_index.json) (786.0 B)

- [stable-diffusion-2-1-unclip-small/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/scheduler/scheduler_config.json) (501.0 B)

- [stable-diffusion-2-1-unclip-small/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/text_encoder/config.json) (671.0 B)

- [stable-diffusion-2-1-unclip-small/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/text_encoder/model_state.pdparams) (1.3 GB)

- [stable-diffusion-2-1-unclip-small/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-2-1-unclip-small/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/tokenizer/special_tokens_map.json) (377.0 B)

- [stable-diffusion-2-1-unclip-small/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/tokenizer/tokenizer_config.json) (806.0 B)

- [stable-diffusion-2-1-unclip-small/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-2-1-unclip-small/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/unet/config.json) (1.5 KB)

- [stable-diffusion-2-1-unclip-small/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/unet/model_state.pdparams) (3.2 GB)

- [stable-diffusion-2-1-unclip-small/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/vae/config.json) (842.0 B)

- [stable-diffusion-2-1-unclip-small/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip-small/vae/model_state.pdparams) (319.1 MB)

- [stable-diffusion-2-1-unclip/feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/feature_extractor/preprocessor_config.json) (466.0 B)

- [stable-diffusion-2-1-unclip/image_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/image_encoder/config.json) (634.0 B)

- [stable-diffusion-2-1-unclip/image_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/image_encoder/model_state.pdparams) (2.4 GB)

- [stable-diffusion-2-1-unclip/image_noising_scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/image_noising_scheduler/scheduler_config.json) (455.0 B)

- [stable-diffusion-2-1-unclip/image_normalizer/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/image_normalizer/config.json) (314.0 B)

- [stable-diffusion-2-1-unclip/image_normalizer/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/image_normalizer/model_state.pdparams) (8.3 KB)

- [stable-diffusion-2-1-unclip/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/model_index.json) (786.0 B)

- [stable-diffusion-2-1-unclip/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/scheduler/scheduler_config.json) (501.0 B)

- [stable-diffusion-2-1-unclip/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/text_encoder/config.json) (671.0 B)

- [stable-diffusion-2-1-unclip/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/text_encoder/model_state.pdparams) (1.3 GB)

- [stable-diffusion-2-1-unclip/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-2-1-unclip/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/tokenizer/special_tokens_map.json) (377.0 B)

- [stable-diffusion-2-1-unclip/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/tokenizer/tokenizer_config.json) (800.0 B)

- [stable-diffusion-2-1-unclip/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-2-1-unclip/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/unet/config.json) (1.5 KB)

- [stable-diffusion-2-1-unclip/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/unet/model_state.pdparams) (3.2 GB)

- [stable-diffusion-2-1-unclip/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/vae/config.json) (836.0 B)

- [stable-diffusion-2-1-unclip/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1-unclip/vae/model_state.pdparams) (319.1 MB)

- [stable-diffusion-2-1/feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/feature_extractor/preprocessor_config.json) (518.0 B)

- [stable-diffusion-2-1/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/model_index.json) (772.0 B)

- [stable-diffusion-2-1/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/scheduler/scheduler_config.json) (535.0 B)

- [stable-diffusion-2-1/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/text_encoder/config.json) (621.0 B)

- [stable-diffusion-2-1/text_encoder/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/text_encoder/model.safetensors) (1.3 GB)

- [stable-diffusion-2-1/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-2-1/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/tokenizer/special_tokens_map.json) (431.0 B)

- [stable-diffusion-2-1/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/tokenizer/tokenizer_config.json) (1.0 KB)

- [stable-diffusion-2-1/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-2-1/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/unet/config.json) (1.9 KB)

- [stable-diffusion-2-1/unet/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/unet/diffusion_paddle_model.safetensors) (3.2 GB)

- [stable-diffusion-2-1/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/vae/config.json) (982.0 B)

- [stable-diffusion-2-1/vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-1/vae/diffusion_paddle_model.safetensors) (319.1 MB)

- [stable-diffusion-2-base/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/config.json) (560.0 B)

- [stable-diffusion-2-base/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/model_config.json) (560.0 B)

- [stable-diffusion-2-base/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/model_index.json) (538.0 B)

- [stable-diffusion-2-base/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/scheduler/scheduler_config.json) (309.0 B)

- [stable-diffusion-2-base/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/text_encoder/config.json) (738.0 B)

- [stable-diffusion-2-base/text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/text_encoder/model_config.json) (237.0 B)

- [stable-diffusion-2-base/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/text_encoder/model_state.pdparams) (1.3 GB)

- [stable-diffusion-2-base/tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/tokenizer/added_tokens.json) (2.0 B)

- [stable-diffusion-2-base/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/tokenizer/merges.txt) (512.4 KB)

- [stable-diffusion-2-base/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/tokenizer/special_tokens_map.json) (668.0 B)

- [stable-diffusion-2-base/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/tokenizer/tokenizer_config.json) (870.0 B)

- [stable-diffusion-2-base/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/tokenizer/vocab.json) (842.1 KB)

- [stable-diffusion-2-base/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/unet/config.json) (845.0 B)

- [stable-diffusion-2-base/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/unet/model_state.pdparams) (3.2 GB)

- [stable-diffusion-2-base/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/vae/config.json) (548.0 B)

- [stable-diffusion-2-base/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base/vae/model_state.pdparams) (319.1 MB)

- [stable-diffusion-2-base@fastdeploy/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/model_index.json) (642.0 B)

- [stable-diffusion-2-base@fastdeploy/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/scheduler/scheduler_config.json) (310.0 B)

- [stable-diffusion-2-base@fastdeploy/text_encoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/text_encoder/inference.pdiparams) (1.3 GB)

- [stable-diffusion-2-base@fastdeploy/text_encoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/text_encoder/inference.pdiparams.info) (35.9 KB)

- [stable-diffusion-2-base@fastdeploy/text_encoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/text_encoder/inference.pdmodel) (763.2 KB)

- [stable-diffusion-2-base@fastdeploy/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/tokenizer/merges.txt) (512.4 KB)

- [stable-diffusion-2-base@fastdeploy/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/tokenizer/special_tokens_map.json) (466.0 B)

- [stable-diffusion-2-base@fastdeploy/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/tokenizer/tokenizer_config.json) (786.0 B)

- [stable-diffusion-2-base@fastdeploy/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/tokenizer/vocab.json) (842.1 KB)

- [stable-diffusion-2-base@fastdeploy/unet/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/unet/inference.pdiparams) (3.2 GB)

- [stable-diffusion-2-base@fastdeploy/unet/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/unet/inference.pdiparams.info) (65.6 KB)

- [stable-diffusion-2-base@fastdeploy/unet/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/unet/inference.pdmodel) (2.4 MB)

- [stable-diffusion-2-base@fastdeploy/vae_decoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/vae_decoder/inference.pdiparams) (188.8 MB)

- [stable-diffusion-2-base@fastdeploy/vae_decoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/vae_decoder/inference.pdiparams.info) (21.4 KB)

- [stable-diffusion-2-base@fastdeploy/vae_decoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/vae_decoder/inference.pdmodel) (302.1 KB)

- [stable-diffusion-2-base@fastdeploy/vae_encoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/vae_encoder/inference.pdiparams) (130.3 MB)

- [stable-diffusion-2-base@fastdeploy/vae_encoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/vae_encoder/inference.pdiparams.info) (21.4 KB)

- [stable-diffusion-2-base@fastdeploy/vae_encoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-base@fastdeploy/vae_encoder/inference.pdmodel) (278.2 KB)

- [stable-diffusion-2-depth/depth_estimator/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/depth_estimator/config.json) (10.0 KB)

- [stable-diffusion-2-depth/depth_estimator/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/depth_estimator/model_state.pdparams) (466.9 MB)

- [stable-diffusion-2-depth/feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/feature_extractor/preprocessor_config.json) (381.0 B)

- [stable-diffusion-2-depth/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/model_index.json) (588.0 B)

- [stable-diffusion-2-depth/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/scheduler/scheduler_config.json) (342.0 B)

- [stable-diffusion-2-depth/text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/text_encoder/model_config.json) (262.0 B)

- [stable-diffusion-2-depth/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/text_encoder/model_state.pdparams) (1.3 GB)

- [stable-diffusion-2-depth/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-2-depth/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/tokenizer/special_tokens_map.json) (377.0 B)

- [stable-diffusion-2-depth/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/tokenizer/tokenizer_config.json) (839.0 B)

- [stable-diffusion-2-depth/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-2-depth/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/unet/config.json) (1.0 KB)

- [stable-diffusion-2-depth/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/unet/model_state.pdparams) (3.2 GB)

- [stable-diffusion-2-depth/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/vae/config.json) (550.0 B)

- [stable-diffusion-2-depth/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-depth/vae/model_state.pdparams) (319.1 MB)

- [stable-diffusion-2-inpainting/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/config.json) (560.0 B)

- [stable-diffusion-2-inpainting/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/model_config.json) (560.0 B)

- [stable-diffusion-2-inpainting/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/model_index.json) (539.0 B)

- [stable-diffusion-2-inpainting/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/scheduler/scheduler_config.json) (310.0 B)

- [stable-diffusion-2-inpainting/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/text_encoder/config.json) (637.0 B)

- [stable-diffusion-2-inpainting/text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/text_encoder/model_config.json) (262.0 B)

- [stable-diffusion-2-inpainting/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/text_encoder/model_state.pdparams) (1.3 GB)

- [stable-diffusion-2-inpainting/tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/tokenizer/added_tokens.json) (2.0 B)

- [stable-diffusion-2-inpainting/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-2-inpainting/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/tokenizer/special_tokens_map.json) (377.0 B)

- [stable-diffusion-2-inpainting/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/tokenizer/tokenizer_config.json) (799.0 B)

- [stable-diffusion-2-inpainting/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-2-inpainting/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/unet/config.json) (907.0 B)

- [stable-diffusion-2-inpainting/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/unet/model_state.pdparams) (3.2 GB)

- [stable-diffusion-2-inpainting/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/vae/config.json) (549.0 B)

- [stable-diffusion-2-inpainting/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting/vae/model_state.pdparams) (319.1 MB)

- [stable-diffusion-2-inpainting@fastdeploy/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/model_index.json) (645.0 B)

- [stable-diffusion-2-inpainting@fastdeploy/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/scheduler/scheduler_config.json) (310.0 B)

- [stable-diffusion-2-inpainting@fastdeploy/text_encoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/text_encoder/inference.pdiparams) (1.3 GB)

- [stable-diffusion-2-inpainting@fastdeploy/text_encoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/text_encoder/inference.pdiparams.info) (35.7 KB)

- [stable-diffusion-2-inpainting@fastdeploy/text_encoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/text_encoder/inference.pdmodel) (762.8 KB)

- [stable-diffusion-2-inpainting@fastdeploy/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-2-inpainting@fastdeploy/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/tokenizer/special_tokens_map.json) (377.0 B)

- [stable-diffusion-2-inpainting@fastdeploy/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/tokenizer/tokenizer_config.json) (725.0 B)

- [stable-diffusion-2-inpainting@fastdeploy/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-2-inpainting@fastdeploy/unet/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/unet/inference.pdiparams) (3.2 GB)

- [stable-diffusion-2-inpainting@fastdeploy/unet/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/unet/inference.pdiparams.info) (65.6 KB)

- [stable-diffusion-2-inpainting@fastdeploy/unet/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/unet/inference.pdmodel) (2.4 MB)

- [stable-diffusion-2-inpainting@fastdeploy/vae_decoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/vae_decoder/inference.pdiparams) (188.8 MB)

- [stable-diffusion-2-inpainting@fastdeploy/vae_decoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/vae_decoder/inference.pdiparams.info) (21.6 KB)

- [stable-diffusion-2-inpainting@fastdeploy/vae_decoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/vae_decoder/inference.pdmodel) (302.4 KB)

- [stable-diffusion-2-inpainting@fastdeploy/vae_encoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/vae_encoder/inference.pdiparams) (130.3 MB)

- [stable-diffusion-2-inpainting@fastdeploy/vae_encoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/vae_encoder/inference.pdiparams.info) (21.6 KB)

- [stable-diffusion-2-inpainting@fastdeploy/vae_encoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2-inpainting@fastdeploy/vae_encoder/inference.pdmodel) (278.4 KB)

- [stable-diffusion-2/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/config.json) (560.0 B)

- [stable-diffusion-2/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/model_config.json) (560.0 B)

- [stable-diffusion-2/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/model_index.json) (538.0 B)

- [stable-diffusion-2/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/model_state.pdparams) (1.3 GB)

- [stable-diffusion-2/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/scheduler/scheduler_config.json) (346.0 B)

- [stable-diffusion-2/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/text_encoder/config.json) (560.0 B)

- [stable-diffusion-2/text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/text_encoder/model_config.json) (560.0 B)

- [stable-diffusion-2/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/text_encoder/model_state.pdparams) (1.3 GB)

- [stable-diffusion-2/tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/tokenizer/added_tokens.json) (2.0 B)

- [stable-diffusion-2/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/tokenizer/merges.txt) (512.4 KB)

- [stable-diffusion-2/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/tokenizer/special_tokens_map.json) (668.0 B)

- [stable-diffusion-2/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/tokenizer/tokenizer_config.json) (870.0 B)

- [stable-diffusion-2/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/tokenizer/vocab.json) (842.1 KB)

- [stable-diffusion-2/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/unet/config.json) (907.0 B)

- [stable-diffusion-2/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/unet/model_state.pdparams) (3.2 GB)

- [stable-diffusion-2/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/vae/config.json) (549.0 B)

- [stable-diffusion-2/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2/vae/model_state.pdparams) (319.1 MB)

- [stable-diffusion-2@fastdeploy/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/model_index.json) (642.0 B)

- [stable-diffusion-2@fastdeploy/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/scheduler/scheduler_config.json) (347.0 B)

- [stable-diffusion-2@fastdeploy/text_encoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/text_encoder/inference.pdiparams) (1.3 GB)

- [stable-diffusion-2@fastdeploy/text_encoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/text_encoder/inference.pdiparams.info) (35.9 KB)

- [stable-diffusion-2@fastdeploy/text_encoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/text_encoder/inference.pdmodel) (763.2 KB)

- [stable-diffusion-2@fastdeploy/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/tokenizer/merges.txt) (512.4 KB)

- [stable-diffusion-2@fastdeploy/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/tokenizer/special_tokens_map.json) (466.0 B)

- [stable-diffusion-2@fastdeploy/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/tokenizer/tokenizer_config.json) (781.0 B)

- [stable-diffusion-2@fastdeploy/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/tokenizer/vocab.json) (842.1 KB)

- [stable-diffusion-2@fastdeploy/unet/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/unet/inference.pdiparams) (3.2 GB)

- [stable-diffusion-2@fastdeploy/unet/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/unet/inference.pdiparams.info) (65.6 KB)

- [stable-diffusion-2@fastdeploy/unet/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/unet/inference.pdmodel) (2.4 MB)

- [stable-diffusion-2@fastdeploy/vae_decoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/vae_decoder/inference.pdiparams) (188.8 MB)

- [stable-diffusion-2@fastdeploy/vae_decoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/vae_decoder/inference.pdiparams.info) (21.4 KB)

- [stable-diffusion-2@fastdeploy/vae_decoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/vae_decoder/inference.pdmodel) (302.1 KB)

- [stable-diffusion-2@fastdeploy/vae_encoder/inference.pdiparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/vae_encoder/inference.pdiparams) (130.3 MB)

- [stable-diffusion-2@fastdeploy/vae_encoder/inference.pdiparams.info](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/vae_encoder/inference.pdiparams.info) (21.4 KB)

- [stable-diffusion-2@fastdeploy/vae_encoder/inference.pdmodel](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-2@fastdeploy/vae_encoder/inference.pdmodel) (278.2 KB)

- [stable-diffusion-3-medium-diffusers/.gitattributes](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/.gitattributes) (1.0 KB)

- [stable-diffusion-3-medium-diffusers/LICENSE](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/LICENSE) (7.2 KB)

- [stable-diffusion-3-medium-diffusers/README.md](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/README.md) (7.3 KB)

- [stable-diffusion-3-medium-diffusers/mmdit.png](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/mmdit.png) (260.2 KB)

- [stable-diffusion-3-medium-diffusers/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/model_index.json) (889.0 B)

- [stable-diffusion-3-medium-diffusers/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/scheduler/scheduler_config.json) (138.0 B)

- [stable-diffusion-3-medium-diffusers/sd3demo.jpg](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/sd3demo.jpg) (7.1 MB)

- [stable-diffusion-3-medium-diffusers/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder/config.json) (598.0 B)

- [stable-diffusion-3-medium-diffusers/text_encoder/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder/model.fp16.safetensors) (235.9 MB)

- [stable-diffusion-3-medium-diffusers/text_encoder/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder/model.safetensors) (471.7 MB)

- [stable-diffusion-3-medium-diffusers/text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_2/config.json) (594.0 B)

- [stable-diffusion-3-medium-diffusers/text_encoder_2/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_2/model.fp16.safetensors) (1.3 GB)

- [stable-diffusion-3-medium-diffusers/text_encoder_2/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_2/model.safetensors) (2.6 GB)

- [stable-diffusion-3-medium-diffusers/text_encoder_3/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_3/config.json) (748.0 B)

- [stable-diffusion-3-medium-diffusers/text_encoder_3/model-00001-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_3/model-00001-of-00002.safetensors) (9.3 GB)

- [stable-diffusion-3-medium-diffusers/text_encoder_3/model-00002-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_3/model-00002-of-00002.safetensors) (8.9 GB)

- [stable-diffusion-3-medium-diffusers/text_encoder_3/model.fp16-00001-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_3/model.fp16-00001-of-00002.safetensors) (4.7 GB)

- [stable-diffusion-3-medium-diffusers/text_encoder_3/model.fp16-00002-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_3/model.fp16-00002-of-00002.safetensors) (4.2 GB)

- [stable-diffusion-3-medium-diffusers/text_encoder_3/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_3/model.safetensors) (18.2 GB)

- [stable-diffusion-3-medium-diffusers/text_encoder_3/model.safetensors.index.fp16.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_3/model.safetensors.index.fp16.json) (20.5 KB)

- [stable-diffusion-3-medium-diffusers/text_encoder_3/model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/text_encoder_3/model.safetensors.index.json) (19.5 KB)

- [stable-diffusion-3-medium-diffusers/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-3-medium-diffusers/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer/special_tokens_map.json) (481.0 B)

- [stable-diffusion-3-medium-diffusers/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer/tokenizer_config.json) (753.0 B)

- [stable-diffusion-3-medium-diffusers/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-3-medium-diffusers/tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer_2/merges.txt) (512.3 KB)

- [stable-diffusion-3-medium-diffusers/tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer_2/special_tokens_map.json) (469.0 B)

- [stable-diffusion-3-medium-diffusers/tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer_2/tokenizer_config.json) (860.0 B)

- [stable-diffusion-3-medium-diffusers/tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer_2/vocab.json) (1.0 MB)

- [stable-diffusion-3-medium-diffusers/tokenizer_3/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer_3/special_tokens_map.json) (2.0 KB)

- [stable-diffusion-3-medium-diffusers/tokenizer_3/spiece.model](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer_3/spiece.model) (773.1 KB)

- [stable-diffusion-3-medium-diffusers/tokenizer_3/tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer_3/tokenizer.json) (2.3 MB)

- [stable-diffusion-3-medium-diffusers/tokenizer_3/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/tokenizer_3/tokenizer_config.json) (15.3 KB)

- [stable-diffusion-3-medium-diffusers/transformer/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/transformer/config.json) (493.0 B)

- [stable-diffusion-3-medium-diffusers/transformer/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/transformer/diffusion_paddle_model.safetensors) (7.8 GB)

- [stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.fp16.safetensors) (3.9 GB)

- [stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.safetensors) (3.9 GB)

- [stable-diffusion-3-medium-diffusers/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/vae/config.json) (920.0 B)

- [stable-diffusion-3-medium-diffusers/vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/vae/diffusion_paddle_model.safetensors) (319.8 MB)

- [stable-diffusion-3-medium-diffusers/vae/diffusion_pytorch_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/vae/diffusion_pytorch_model.fp16.safetensors) (159.9 MB)

- [stable-diffusion-3-medium-diffusers/vae/diffusion_pytorch_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3-medium-diffusers/vae/diffusion_pytorch_model.safetensors) (159.9 MB)

- [stable-diffusion-3.5-medium/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/model_index.json) (856.0 B)

- [stable-diffusion-3.5-medium/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/scheduler/scheduler_config.json) (138.0 B)

- [stable-diffusion-3.5-medium/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder/config.json) (589.0 B)

- [stable-diffusion-3.5-medium/text_encoder/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder/model.safetensors) (235.9 MB)

- [stable-diffusion-3.5-medium/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder/model_state.pdparams) (235.9 MB)

- [stable-diffusion-3.5-medium/text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder_2/config.json) (585.0 B)

- [stable-diffusion-3.5-medium/text_encoder_2/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder_2/model.safetensors) (1.3 GB)

- [stable-diffusion-3.5-medium/text_encoder_3/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder_3/config.json) (705.0 B)

- [stable-diffusion-3.5-medium/text_encoder_3/model-00001-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder_3/model-00001-of-00002.safetensors) (9.3 GB)

- [stable-diffusion-3.5-medium/text_encoder_3/model-00002-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder_3/model-00002-of-00002.safetensors) (1.7 GB)

- [stable-diffusion-3.5-medium/text_encoder_3/model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/text_encoder_3/model.safetensors.index.json) (19.5 KB)

- [stable-diffusion-3.5-medium/tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer/added_tokens.json) (50.0 B)

- [stable-diffusion-3.5-medium/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-3.5-medium/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer/special_tokens_map.json) (553.0 B)

- [stable-diffusion-3.5-medium/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer/tokenizer_config.json) (719.0 B)

- [stable-diffusion-3.5-medium/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-3.5-medium/tokenizer_2/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_2/added_tokens.json) (58.0 B)

- [stable-diffusion-3.5-medium/tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_2/merges.txt) (512.3 KB)

- [stable-diffusion-3.5-medium/tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_2/special_tokens_map.json) (541.0 B)

- [stable-diffusion-3.5-medium/tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_2/tokenizer_config.json) (826.0 B)

- [stable-diffusion-3.5-medium/tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_2/vocab.json) (1.0 MB)

- [stable-diffusion-3.5-medium/tokenizer_3/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_3/added_tokens.json) (2.4 KB)

- [stable-diffusion-3.5-medium/tokenizer_3/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_3/special_tokens_map.json) (2.1 KB)

- [stable-diffusion-3.5-medium/tokenizer_3/spiece.model](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_3/spiece.model) (773.1 KB)

- [stable-diffusion-3.5-medium/tokenizer_3/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/tokenizer_3/tokenizer_config.json) (15.3 KB)

- [stable-diffusion-3.5-medium/transformer/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/transformer/config.json) (611.0 B)

- [stable-diffusion-3.5-medium/transformer/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/transformer/diffusion_paddle_model.safetensors) (4.6 GB)

- [stable-diffusion-3.5-medium/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/vae/config.json) (921.0 B)

- [stable-diffusion-3.5-medium/vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-3.5-medium/vae/diffusion_paddle_model.safetensors) (159.9 MB)

- [stable-diffusion-v1-5/feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/feature_extractor/preprocessor_config.json) (342.0 B)

- [stable-diffusion-v1-5/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/model_index.json) (601.0 B)

- [stable-diffusion-v1-5/safety_checker/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/safety_checker/config.json) (553.0 B)

- [stable-diffusion-v1-5/safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/safety_checker/model_config.json) (614.0 B)

- [stable-diffusion-v1-5/safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/safety_checker/model_state.pdparams) (1.1 GB)

- [stable-diffusion-v1-5/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/scheduler/scheduler_config.json) (342.0 B)

- [stable-diffusion-v1-5/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/text_encoder/config.json) (592.0 B)

- [stable-diffusion-v1-5/text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/text_encoder/model_config.json) (463.0 B)

- [stable-diffusion-v1-5/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/text_encoder/model_state.pdparams) (469.5 MB)

- [stable-diffusion-v1-5/tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/tokenizer/added_tokens.json) (2.0 B)

- [stable-diffusion-v1-5/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/tokenizer/merges.txt) (512.4 KB)

- [stable-diffusion-v1-5/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/tokenizer/special_tokens_map.json) (478.0 B)

- [stable-diffusion-v1-5/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/tokenizer/tokenizer_config.json) (312.0 B)

- [stable-diffusion-v1-5/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/tokenizer/vocab.json) (842.1 KB)

- [stable-diffusion-v1-5/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/unet/config.json) (807.0 B)

- [stable-diffusion-v1-5/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/unet/model_state.pdparams) (3.2 GB)

- [stable-diffusion-v1-5/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/vae/config.json) (610.0 B)

- [stable-diffusion-v1-5/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-v1-5/vae/model_state.pdparams) (319.1 MB)

- [stable-diffusion-x4-upscaler/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/config.json) (560.0 B)

- [stable-diffusion-x4-upscaler/low_res_scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/low_res_scheduler/scheduler_config.json) (301.0 B)

- [stable-diffusion-x4-upscaler/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/model_config.json) (560.0 B)

- [stable-diffusion-x4-upscaler/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/model_index.json) (510.0 B)

- [stable-diffusion-x4-upscaler/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/scheduler/scheduler_config.json) (349.0 B)

- [stable-diffusion-x4-upscaler/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/text_encoder/config.json) (633.0 B)

- [stable-diffusion-x4-upscaler/text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/text_encoder/model_config.json) (262.0 B)

- [stable-diffusion-x4-upscaler/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/text_encoder/model_state.pdparams) (1.3 GB)

- [stable-diffusion-x4-upscaler/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-x4-upscaler/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/tokenizer/special_tokens_map.json) (377.0 B)

- [stable-diffusion-x4-upscaler/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/tokenizer/tokenizer_config.json) (816.0 B)

- [stable-diffusion-x4-upscaler/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-x4-upscaler/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/unet/config.json) (917.0 B)

- [stable-diffusion-x4-upscaler/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/unet/model_state.pdparams) (1.8 GB)

- [stable-diffusion-x4-upscaler/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/vae/config.json) (494.0 B)

- [stable-diffusion-x4-upscaler/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-x4-upscaler/vae/model_state.pdparams) (211.1 MB)

- [stable-diffusion-xl-base-1.0/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/model_index.json) (585.0 B)

- [stable-diffusion-xl-base-1.0/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/scheduler/scheduler_config.json) (549.0 B)

- [stable-diffusion-xl-base-1.0/sd_xl_offset_example-lora_1.0.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/sd_xl_offset_example-lora_1.0.safetensors) (47.3 MB)

- [stable-diffusion-xl-base-1.0/text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/text_encoder/config.json) (611.0 B)

- [stable-diffusion-xl-base-1.0/text_encoder/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/text_encoder/model.fp16.safetensors) (234.7 MB)

- [stable-diffusion-xl-base-1.0/text_encoder/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/text_encoder/model.safetensors) (234.7 MB)

- [stable-diffusion-xl-base-1.0/text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/text_encoder/model_state.pdparams) (234.7 MB)

- [stable-diffusion-xl-base-1.0/text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2/config.json) (621.0 B)

- [stable-diffusion-xl-base-1.0/text_encoder_2/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2/model.fp16.safetensors) (1.3 GB)

- [stable-diffusion-xl-base-1.0/text_encoder_2/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2/model.safetensors) (1.3 GB)

- [stable-diffusion-xl-base-1.0/text_encoder_2/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2/model_state.pdparams) (1.3 GB)

- [stable-diffusion-xl-base-1.0/tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/tokenizer/merges.txt) (512.3 KB)

- [stable-diffusion-xl-base-1.0/tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/tokenizer/special_tokens_map.json) (389.0 B)

- [stable-diffusion-xl-base-1.0/tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/tokenizer/tokenizer_config.json) (846.0 B)

- [stable-diffusion-xl-base-1.0/tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/tokenizer/vocab.json) (1.0 MB)

- [stable-diffusion-xl-base-1.0/tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2/merges.txt) (512.3 KB)

- [stable-diffusion-xl-base-1.0/tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2/special_tokens_map.json) (377.0 B)

- [stable-diffusion-xl-base-1.0/tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2/tokenizer_config.json) (836.0 B)

- [stable-diffusion-xl-base-1.0/tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2/vocab.json) (1.0 MB)

- [stable-diffusion-xl-base-1.0/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/unet/config.json) (1.7 KB)

- [stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model-00001-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model-00001-of-00003.safetensors) (4.6 GB)

- [stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model-00002-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model-00002-of-00003.safetensors) (4.6 GB)

- [stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model-00003-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model-00003-of-00003.safetensors) (318.9 MB)

- [stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model.fp16.safetensors) (4.8 GB)

- [stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/unet/diffusion_paddle_model.safetensors.index.json) (194.8 KB)

- [stable-diffusion-xl-base-1.0/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/unet/model_state.pdparams) (4.8 GB)

- [stable-diffusion-xl-base-1.0/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/vae/config.json) (672.0 B)

- [stable-diffusion-xl-base-1.0/vae/diffusion_paddle_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/vae/diffusion_paddle_model.fp16.safetensors) (159.6 MB)

- [stable-diffusion-xl-base-1.0/vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/vae/diffusion_paddle_model.safetensors) (319.1 MB)

- [stable-diffusion-xl-base-1.0/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-base-1.0/vae/model_state.pdparams) (159.6 MB)

- [stable-diffusion-xl-refiner-1.0/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/model_index.json) (608.0 B)

- [stable-diffusion-xl-refiner-1.0/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/scheduler/scheduler_config.json) (476.0 B)

- [stable-diffusion-xl-refiner-1.0/text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/text_encoder_2/config.json) (621.0 B)

- [stable-diffusion-xl-refiner-1.0/text_encoder_2/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/text_encoder_2/model_state.pdparams) (1.3 GB)

- [stable-diffusion-xl-refiner-1.0/tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/tokenizer_2/merges.txt) (512.3 KB)

- [stable-diffusion-xl-refiner-1.0/tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/tokenizer_2/special_tokens_map.json) (377.0 B)

- [stable-diffusion-xl-refiner-1.0/tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/tokenizer_2/tokenizer_config.json) (832.0 B)

- [stable-diffusion-xl-refiner-1.0/tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/tokenizer_2/vocab.json) (1.0 MB)

- [stable-diffusion-xl-refiner-1.0/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/unet/config.json) (1.9 KB)

- [stable-diffusion-xl-refiner-1.0/unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/unet/model_state.pdparams) (4.2 GB)

- [stable-diffusion-xl-refiner-1.0/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/vae/config.json) (831.0 B)

- [stable-diffusion-xl-refiner-1.0/vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-diffusion-xl-refiner-1.0/vae/model_state.pdparams) (159.6 MB)

- [stable-video-diffusion-img2vid-xt/feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/feature_extractor/preprocessor_config.json) (518.0 B)

- [stable-video-diffusion-img2vid-xt/image_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/image_encoder/config.json) (731.0 B)

- [stable-video-diffusion-img2vid-xt/image_encoder/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/image_encoder/model.fp16.safetensors) (1.2 GB)

- [stable-video-diffusion-img2vid-xt/image_encoder/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/image_encoder/model.safetensors) (1.2 GB)

- [stable-video-diffusion-img2vid-xt/model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/model_index.json) (621.0 B)

- [stable-video-diffusion-img2vid-xt/scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/scheduler/scheduler_config.json) (530.0 B)

- [stable-video-diffusion-img2vid-xt/unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/unet/config.json) (969.0 B)

- [stable-video-diffusion-img2vid-xt/unet/diffusion_paddle_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/unet/diffusion_paddle_model.fp16.safetensors) (5.7 GB)

- [stable-video-diffusion-img2vid-xt/unet/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/unet/diffusion_paddle_model.safetensors) (5.7 GB)

- [stable-video-diffusion-img2vid-xt/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/vae/config.json) (592.0 B)

- [stable-video-diffusion-img2vid-xt/vae/diffusion_paddle_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/vae/diffusion_paddle_model.fp16.safetensors) (372.9 MB)

- [stable-video-diffusion-img2vid-xt/vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid-xt/vae/diffusion_paddle_model.safetensors) (372.9 MB)

- [stable-video-diffusion-img2vid/.gitattributes](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid/.gitattributes) (1.5 KB)

- [stable-video-diffusion-img2vid/vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid/vae/config.json) (609.0 B)

- [stable-video-diffusion-img2vid/vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/stable-video-diffusion-img2vid/vae/diffusion_paddle_model.safetensors) (372.9 MB)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/text_encoder/config.json) (589.0 B)

- [text_encoder/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/text_encoder/model.safetensors) (235.9 MB)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/text_encoder/model_state.pdparams) (235.9 MB)

- [text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/text_encoder_2/config.json) (585.0 B)

- [text_encoder_2/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/text_encoder_2/model.safetensors) (1.3 GB)

- [text_encoder_3/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/text_encoder_3/config.json) (705.0 B)

- [text_encoder_3/model-00002-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/text_encoder_3/model-00002-of-00002.safetensors) (1.7 GB)

- [text_encoder_3/model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/text_encoder_3/model.safetensors.index.json) (19.5 KB)

- [tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer/added_tokens.json) (50.0 B)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer/special_tokens_map.json) (553.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer/tokenizer_config.json) (719.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer/vocab.json) (1.0 MB)

- [tokenizer_2/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_2/added_tokens.json) (58.0 B)

- [tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_2/merges.txt) (512.3 KB)

- [tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_2/special_tokens_map.json) (541.0 B)

- [tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_2/tokenizer_config.json) (826.0 B)

- [tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_2/vocab.json) (1.0 MB)

- [tokenizer_3/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_3/added_tokens.json) (2.4 KB)

- [tokenizer_3/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_3/special_tokens_map.json) (2.1 KB)

- [tokenizer_3/spiece.model](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_3/spiece.model) (773.1 KB)

- [tokenizer_3/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/tokenizer_3/tokenizer_config.json) (15.3 KB)

- [transformer/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/transformer/config.json) (611.0 B)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/vae/config.json) (921.0 B)

- [vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/stabilityai/vae/diffusion_paddle_model.safetensors) (159.9 MB)


[Back to Main](../)