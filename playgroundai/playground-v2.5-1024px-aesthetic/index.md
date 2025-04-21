
# playground-v2.5-1024px-aesthetic
---


## README([From Huggingface](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic))


# Playground v2.5 – 1024px Aesthetic Model

This repository contains a model that generates highly aesthetic images of resolution 1024x1024, as well as portrait and landscape aspect ratios. You can use the model with Hugging Face 🧨 Diffusers.

![![image/png](https://cdn-uploads.huggingface.co/production/uploads/636c0c4eaae2da3c76b8a9a3/HYUUGfU6SOCHsvyeISQ5Y.png)

**Playground v2.5** is a diffusion-based text-to-image generative model, and a successor to [Playground v2](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic).

Playground v2.5 is the state-of-the-art open-source model in aesthetic quality. Our user studies demonstrate that our model outperforms SDXL, Playground v2, PixArt-α, DALL-E 3, and Midjourney 5.2.

For details on the development and training of our model, please refer to our [blog post](https://blog.playgroundai.com/playground-v2-5/) and [technical report](https://marketing-cdn.playground.com/research/pgv2.5_compressed.pdf).

### Model Description
- **Developed by:** [Playground](https://playground.com)
- **Model type:** Diffusion-based text-to-image generative model
- **License:** [Playground v2.5 Community License](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md)
- **Summary:** This model generates images based on text prompts. It is a Latent Diffusion Model that uses two fixed, pre-trained text encoders (OpenCLIP-ViT/G and CLIP-ViT/L). It follows the same architecture as [Stable Diffusion XL](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl).

### Using the model with 🧨 Diffusers

Install diffusers >= 0.27.0 and the relevant dependencies.

```
pip install diffusers>=0.27.0
pip install transformers accelerate safetensors
```

**Notes:**
- The pipeline uses the `EDMDPMSolverMultistepScheduler` scheduler by default, for crisper fine details. It's an [EDM formulation](https://arxiv.org/abs/2206.00364) of the DPM++ 2M Karras scheduler. `guidance_scale=3.0` is a good default for this scheduler.
- The pipeline also supports the `EDMEulerScheduler` scheduler. It's an [EDM formulation](https://arxiv.org/abs/2206.00364) of the Euler scheduler. `guidance_scale=5.0` is a good default for this scheduler.

Then, run the following snippet:

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    dtype=paddle.float16,
    variant="fp16",
)

# # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
# from diffusers import EDMDPMSolverMultistepScheduler
# pipe.scheduler = EDMDPMSolverMultistepScheduler()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
```

### Using the model with Automatic1111/ComfyUI

Support coming soon. We will update this model card with instructions when ready.

### User Studies

This model card only provides a brief summary of our user study results. For extensive details on how we perform user studies, please check out our [technical report](https://marketing-cdn.playground.com/research/pgv2.5_compressed.pdf).

We conducted studies to measure overall aesthetic quality, as well as for the specific areas we aimed to improve with Playground v2.5, namely multi aspect ratios and human preference alignment.

#### Comparison to State-of-the-Art

![![image/png](https://cdn-uploads.huggingface.co/production/uploads/63855d851769b7c4b10e1f76/V7LFNzgoQJnL__ndU0CnE.png)

The aesthetic quality of Playground v2.5 dramatically outperforms the current state-of-the-art open source models SDXL and PIXART-α, as well as Playground v2. Because the performance differential between Playground V2.5 and SDXL was so large, we also tested our aesthetic quality against world-class closed-source models like DALL-E 3 and Midjourney 5.2, and found that Playground v2.5 outperforms them as well.

#### Multi Aspect Ratios

![![image/png](https://cdn-uploads.huggingface.co/production/uploads/636c0c4eaae2da3c76b8a9a3/xMB0r-CmR3N6dABFlcV71.png)

Similarly, for multi aspect ratios, we outperform SDXL by a large margin.

#### Human Preference Alignment on People-related images

![![image/png](https://cdn-uploads.huggingface.co/production/uploads/636c0c4eaae2da3c76b8a9a3/7c-8Stw52OsNtUjse8Slv.png)

Next, we benchmark Playground v2.5 specifically on people-related images, to test Human Preference Alignment. We compared Playground v2.5 against two commonly-used baseline models: SDXL and RealStock v2, a community fine-tune of SDXL that was trained on a realistic people dataset.

Playground v2.5 outperforms both baselines by a large margin.

### MJHQ-30K Benchmark

![![image/png](https://cdn-uploads.huggingface.co/production/uploads/636c0c4eaae2da3c76b8a9a3/7tyYDPGUtokh-k18XDSte.png)

| Model                                 | Overall FID   |
| ------------------------------------- | ----- |
| SDXL-1-0-refiner                      | 9.55  |
| [playground-v2-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic)        | 7.07  |
| [playground-v2.5-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) | **4.48** |

Lastly, we report metrics using our MJHQ-30K benchmark which we [open-sourced](https://huggingface.co/datasets/playgroundai/MJHQ-30K) with the v2 release. We report both the overall FID and per category FID. All FID metrics are computed at resolution 1024x1024. Our results show that Playground v2.5 outperforms both Playground v2 and SDXL in overall FID and all category FIDs, especially in the people and fashion categories. This is in line with the results of the user study, which indicates a correlation between human preferences and the FID score of the MJHQ-30K benchmark.

### How to cite us

```
@misc{li2024playground,
      title={Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation}, 
      author={Daiqing Li and Aleks Kamko and Ehsan Akhgari and Ali Sabet and Linmiao Xu and Suhail Doshi},
      year={2024},
      eprint={2402.17245},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/README.md) (6.3 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/model_index.json) (803.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/scheduler/scheduler_config.json) (494.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/text_encoder/config.json) (606.0 B)

- [text_encoder/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/text_encoder/model.fp16.safetensors) (234.7 MB)

- [text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/text_encoder_2/config.json) (616.0 B)

- [text_encoder_2/model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/text_encoder_2/model.fp16.safetensors) (1.3 GB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/tokenizer/special_tokens_map.json) (479.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/tokenizer/tokenizer_config.json) (749.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/tokenizer/vocab.json) (1.0 MB)

- [tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/tokenizer_2/merges.txt) (512.3 KB)

- [tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/tokenizer_2/special_tokens_map.json) (377.0 B)

- [tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/tokenizer_2/tokenizer_config.json) (856.0 B)

- [tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/tokenizer_2/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/unet/config.json) (1.8 KB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/vae/config.json) (879.0 B)

- [vae/diffusion_paddle_model.fp16.safetensors](https://paddlenlp.bj.bcebos.com/models/community/playgroundai/playground-v2.5-1024px-aesthetic/vae/diffusion_paddle_model.fp16.safetensors) (159.6 MB)


[Back to Main](../../)