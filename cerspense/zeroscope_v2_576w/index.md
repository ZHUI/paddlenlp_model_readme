
# zeroscope_v2_576w
---


## README([From Huggingface](https://huggingface.co/cerspense/zeroscope_v2_576w))

---
pipeline_tag: text-to-video
license: cc-by-nc-4.0
---

![![model example](https://i.imgur.com/1mrNnh8.png)

# zeroscope_v2 576w
A watermark-free Modelscope-based video model optimized for producing high-quality 16:9 compositions and a smooth video output. This model was trained from the [original weights](https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis) using 9,923 clips and 29,769 tagged frames at 24 frames, 576x320 resolution.<br />
zeroscope_v2_567w is specifically designed for upscaling with [zeroscope_v2_XL](https://huggingface.co/cerspense/zeroscope_v2_XL) using vid2vid in the [1111 text2video](https://github.com/kabachuha/sd-webui-text2video) extension by [kabachuha](https://github.com/kabachuha). Leveraging this model as a preliminary step allows for superior overall compositions at higher resolutions in zeroscope_v2_XL, permitting faster exploration in 576x320 before transitioning to a high-resolution render. See some [example outputs](https://www.youtube.com/watch?v=HO3APT_0UA4) that have been upscaled to 1024x576 using zeroscope_v2_XL. (courtesy of [dotsimulate](https://www.instagram.com/dotsimulate/))<br />

zeroscope_v2_576w uses 7.9gb of vram when rendering 30 frames at 576x320

### Using it with the 1111 text2video extension

1. Download files in the zs2_576w folder.
2. Replace the respective files in the 'stable-diffusion-webui\models\ModelScope\t2v' directory.

### Upscaling recommendations

For upscaling, it's recommended to use [zeroscope_v2_XL](https://huggingface.co/cerspense/zeroscope_v2_XL) via vid2vid in the 1111 extension. It works best at 1024x576 with a denoise strength between 0.66 and 0.85. Remember to use the same prompt that was used to generate the original clip. <br />

### Usage in 🧨 Diffusers

Let's first install the libraries required:

```bash
$ pip install diffusers transformers accelerate torch
```

Now, generate a video:

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", dtype=paddle.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Darth Vader is surfing on waves"
video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
video_path = export_to_video(video_frames)
```

Here are some results:

<table>
    <tr>
        Darth vader is surfing on waves.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/darthvader_cerpense.gif"
            alt="Darth vader surfing in waves."
            style="width: 576;" />
        </center></td>
    </tr>
</table>

### Known issues

Lower resolutions or fewer frames could lead to suboptimal output. <br />

Thanks to [camenduru](https://github.com/camenduru), [kabachuha](https://github.com/kabachuha), [ExponentialML](https://github.com/ExponentialML), [dotsimulate](https://www.instagram.com/dotsimulate/), [VANYA](https://twitter.com/veryVANYA), [polyware](https://twitter.com/polyware_ai), [tin2tin](https://github.com/tin2tin)<br />



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/README.md) (3.2 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/model_index.json) (407.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/scheduler/scheduler_config.json) (530.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/text_encoder/config.json) (655.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/text_encoder/model_state.pdparams) (649.3 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/tokenizer/special_tokens_map.json) (377.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/tokenizer/tokenizer_config.json) (826.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/unet/config.json) (845.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/unet/model_state.pdparams) (2.6 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/vae/config.json) (814.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/cerspense/zeroscope_v2_576w/vae/model_state.pdparams) (159.6 MB)


[Back to Main](../../)