
# mo-di-diffusion
---


## README([From Huggingface](https://huggingface.co/nitrosocke/mo-di-diffusion))


**Mo Di Diffusion**

This is the fine-tuned Stable Diffusion 1.5 model trained on screenshots from a popular animation studio.
Use the tokens **_modern disney style_** in your prompts for the effect.

**If you enjoy my work, please consider supporting me** 
[![![Become A Patreon](https://badgen.net/badge/become/a%20patron/F96854)](https://patreon.com/user?u=79196446)

**Videogame Characters rendered with the model:**
![![Videogame Samples](https://huggingface.co/nitrosocke/mo-di-diffusion/resolve/main/modi-samples-01s.jpg)
**Animal Characters rendered with the model:**
![![Animal Samples](https://huggingface.co/nitrosocke/mo-di-diffusion/resolve/main/modi-samples-02s.jpg)
**Cars and Landscapes rendered with the model:**
![![Misc. Samples](https://huggingface.co/nitrosocke/mo-di-diffusion/resolve/main/modi-samples-03s.jpg)
#### Prompt and settings for Lara Croft:
**modern disney lara croft**
_Steps: 50, Sampler: Euler a, CFG scale: 7, Seed: 3940025417, Size: 512x768_

#### Prompt and settings for the Lion:
**modern disney (baby lion) Negative prompt: person human**
_Steps: 50, Sampler: Euler a, CFG scale: 7, Seed: 1355059992, Size: 512x512_

This model was trained using the diffusers based dreambooth training by ShivamShrirao using prior-preservation loss and the _train-text-encoder_ flag in 9.000 steps.

### 🧨 Diffusers

This model can be used just like any other Stable Diffusion model. For more information,
please have a look at the [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion).

You can also export the model to [ONNX](https://huggingface.co/docs/diffusers/optimization/onnx), [MPS](https://huggingface.co/docs/diffusers/optimization/mps) and/or [FLAX/JAX]().

```python
from diffusers import StableDiffusionPipeline
import paddle

model_id = "nitrosocke/mo-di-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=paddle.float16)
pipe = pipe

prompt = "a magical princess with golden hair, modern disney style"
image = pipe(prompt).images[0]

image.save("./magical_princess.png")
```

# Gradio & Colab

We also support a [Gradio](https://github.com/gradio-app/gradio) Web UI and Colab with Diffusers to run fine-tuned Stable Diffusion models:
[![![Open In Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/anzorq/finetuned_diffusion)
[![![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j5YvfMZoGdDGdj3O3xRU1m4ujKYsElZO?usp=sharing)

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/README.md) (3.6 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/feature_extractor/preprocessor_config.json) (407.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/model_index.json) (616.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/safety_checker/model_config.json) (368.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/scheduler/scheduler_config.json) (318.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/text_encoder/model_config.json) (267.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/tokenizer/tokenizer_config.json) (769.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/unet/config.json) (1.0 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/vae/config.json) (670.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/mo-di-diffusion/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)