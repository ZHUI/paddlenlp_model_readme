
# redshift-diffusion
---


## README([From Huggingface](https://huggingface.co/nitrosocke/redshift-diffusion))

---
language:
- en
license: creativeml-openrail-m
thumbnail: "https://huggingface.co/nitrosocke/redshift-diffusion/resolve/main/images/redshift-diffusion-samples-01s.jpg"
tags:
- stable-diffusion
- text-to-image
- image-to-image

---
### Redshift Diffusion

This is the fine-tuned Stable Diffusion model trained on high resolution 3D artworks.
Use the tokens **_redshift style_** in your prompts for the effect.

**The name:** I used Cinema4D for a very long time as my go-to modeling software and always liked the redshift render it came with. That is why I was very sad to see the bad results base SD has connected with its token. This is my attempt at fixing that and showing my passion for this render engine.

**If you enjoy my work and want to test new models before release, please consider supporting me**
[![![Become A Patreon](https://badgen.net/badge/become/a%20patron/F96854)](https://patreon.com/user?u=79196446)

**Characters rendered with the model:**
![![Videogame Samples](https://huggingface.co/nitrosocke/redshift-diffusion/resolve/main/images/redshift-diffusion-samples-01s.jpg)
**Cars and Landscapes rendered with the model:**
![![Misc. Samples](https://huggingface.co/nitrosocke/redshift-diffusion/resolve/main/images/redshift-diffusion-samples-02s.jpg)

#### Prompt and settings for Tony Stark:
**(redshift style) robert downey jr as ironman Negative prompt: glasses helmet**
_Steps: 40, Sampler: DPM2 Karras, CFG scale: 7, Seed: 908018284, Size: 512x704_

#### Prompt and settings for the Ford Mustang:
**redshift style Ford Mustang**
_Steps: 20, Sampler: DPM2 Karras, CFG scale: 7, Seed: 579593863, Size: 704x512_

This model was trained using the diffusers based dreambooth training by ShivamShrirao using prior-preservation loss and the _train-text-encoder_ flag in 11.000 steps.

### Gradio

We support a [Gradio](https://github.com/gradio-app/gradio) Web UI run redshift-diffusion:
[![![Open In Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/nitrosocke/Redshift-Diffusion-Demo)

### 🧨 Diffusers

This model can be used just like any other Stable Diffusion model. For more information,
please have a look at the [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion).

You can also export the model to [ONNX](https://huggingface.co/docs/diffusers/optimization/onnx), [MPS](https://huggingface.co/docs/diffusers/optimization/mps) and/or [FLAX/JAX]().

```python
from diffusers import StableDiffusionPipeline
import paddle

model_id = "nitrosocke/redshift-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=paddle.float16)
pipe = pipe

prompt = "redshift style magical princess with golden hair"
image = pipe(prompt).images[0]

image.save("./magical_princess.png")
```

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/README.md) (3.8 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/feature_extractor/preprocessor_config.json) (407.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/model_index.json) (616.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/safety_checker/model_config.json) (368.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/scheduler/scheduler_config.json) (318.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/text_encoder/model_config.json) (267.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/tokenizer/tokenizer_config.json) (772.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/unet/config.json) (1.0 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/vae/config.json) (673.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/redshift-diffusion/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)