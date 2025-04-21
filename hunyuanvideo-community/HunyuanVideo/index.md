
# HunyuanVideo
---


## README([From Huggingface](https://huggingface.co/hunyuanvideo-community/HunyuanVideo))

---
base_model:
- tencent/HunyuanVideo
library_name: diffusers
---

Unofficial community fork for Diffusers-format weights on [`tencent/HunyuanVideo`](https://huggingface.co/tencent/HunyuanVideo).

### Using Diffusers

HunyuanVideo can be used directly from Diffusers. Install the latest version of Diffusers.

```python
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", dtype=paddle.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, dtype=paddle.float16)

# Enable memory savings
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
```

Refer to the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuan_video) for more information.





## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/README.md) (1.1 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/config.json) (38.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/model_index.json) (563.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/scheduler/scheduler_config.json) (419.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/text_encoder/config.json) (1.1 KB)

- [text_encoder/model_state-00001-of-00002.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/text_encoder/model_state-00001-of-00002.pdparams) (9.3 GB)

- [text_encoder/model_state-00002-of-00002.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/text_encoder/model_state-00002-of-00002.pdparams) (4.7 GB)

- [text_encoder/model_state.pdparams.index.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/text_encoder/model_state.pdparams.index.json) (22.5 KB)

- [text_encoder_2/config.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/text_encoder_2/config.json) (4.4 KB)

- [text_encoder_2/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/text_encoder_2/model_state.pdparams) (1.6 GB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/tokenizer/special_tokens_map.json) (577.0 B)

- [tokenizer/tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/tokenizer/tokenizer.json) (16.4 MB)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/tokenizer/tokenizer_config.json) (50.5 KB)

- [tokenizer_2/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/tokenizer_2/merges.txt) (512.4 KB)

- [tokenizer_2/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/tokenizer_2/special_tokens_map.json) (478.0 B)

- [tokenizer_2/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/tokenizer_2/tokenizer_config.json) (194.0 B)

- [tokenizer_2/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/tokenizer_2/vocab.json) (842.1 KB)

- [transformer/config.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/transformer/config.json) (543.0 B)

- [transformer/diffusion_paddle_model-00001-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/transformer/diffusion_paddle_model-00001-of-00003.safetensors) (9.3 GB)

- [transformer/diffusion_paddle_model-00002-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/transformer/diffusion_paddle_model-00002-of-00003.safetensors) (9.3 GB)

- [transformer/diffusion_paddle_model-00003-of-00003.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/transformer/diffusion_paddle_model-00003-of-00003.safetensors) (5.3 GB)

- [transformer/diffusion_paddle_model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/transformer/diffusion_paddle_model.safetensors.index.json) (128.8 KB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/vae/config.json) (751.0 B)

- [vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/hunyuanvideo-community/HunyuanVideo/vae/diffusion_paddle_model.safetensors) (940.3 MB)


[Back to Main](../../)