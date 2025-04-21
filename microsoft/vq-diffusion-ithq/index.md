
# vq-diffusion-ithq
---


## README([From Huggingface](https://huggingface.co/microsoft/vq-diffusion-ithq))


# VQ Diffusion

* [Paper](https://arxiv.org/abs/2205.16007.pdf)

* [Original Repo](https://github.com/microsoft/VQ-Diffusion)

* **Authors**: Shuyang Gu, Dong Chen, et al.


```python
#!pip install diffusers[torch] transformers
import torch
from diffusers import VQDiffusionPipeline

pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

output = pipeline("teddy bear playing in the pool", truncation_rate=1.0)

image = output.images[0]
image.save("./teddy_bear.png")
```

![![img](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/vq_diffusion_fp16.png)

**Contribution**: This model was contribution by [williamberman](https://huggingface.co/williamberman) in [VQ-diffusion](https://github.com/huggingface/diffusers/pull/658).




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/README.md) (850.0 B)

- [learned_classifier_free_sampling_embeddings/config.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/learned_classifier_free_sampling_embeddings/config.json) (157.0 B)

- [learned_classifier_free_sampling_embeddings/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/learned_classifier_free_sampling_embeddings/model_state.pdparams) (154.3 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/model_index.json) (534.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/scheduler/scheduler_config.json) (248.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/text_encoder/model_config.json) (266.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/text_encoder/model_state.pdparams) (241.0 MB)

- [tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/tokenizer/added_tokens.json) (2.0 B)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/tokenizer/special_tokens_map.json) (377.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/tokenizer/tokenizer_config.json) (799.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/tokenizer/vocab.json) (1.0 MB)

- [transformer/config.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/transformer/config.json) (396.0 B)

- [transformer/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/transformer/model_state.pdparams) (5.1 GB)

- [vqvae/config.json](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/vqvae/config.json) (603.0 B)

- [vqvae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/microsoft/vq-diffusion-ithq/vqvae/model_state.pdparams) (244.2 MB)


[Back to Main](../../)