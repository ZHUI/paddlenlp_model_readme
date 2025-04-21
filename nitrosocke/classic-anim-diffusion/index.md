
# classic-anim-diffusion
---


## README([From Huggingface](https://huggingface.co/nitrosocke/classic-anim-diffusion))


### Classic Animation Diffusion

This is the fine-tuned Stable Diffusion model trained on screenshots from a popular animation studio.
Use the tokens **_classic disney style_** in your prompts for the effect.

**If you enjoy my work, please consider supporting me** 
[![![Become A Patreon](https://badgen.net/badge/become/a%20patron/F96854)](https://patreon.com/user?u=79196446)

**Characters rendered with the model:**
![![Videogame Samples](https://huggingface.co/nitrosocke/classic-anim-diffusion/resolve/main/clanim-samples-01s.jpg)
**Animals rendered with the model:**
![![Animal Samples](https://huggingface.co/nitrosocke/classic-anim-diffusion/resolve/main/clanim-samples-02s.jpg)
**Cars and Landscapes rendered with the model:**
![![Misc. Samples](https://huggingface.co/nitrosocke/classic-anim-diffusion/resolve/main/clanim-samples-03s.jpg)

### 🧨 Diffusers

This model can be used just like any other Stable Diffusion model. For more information,
please have a look at the [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion).

You can also export the model to [ONNX](https://huggingface.co/docs/diffusers/optimization/onnx), [MPS](https://huggingface.co/docs/diffusers/optimization/mps) and/or [FLAX/JAX]().

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "nitrosocke/classic-anim-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "classic disney style magical princess with golden hair"
image = pipe(prompt).images[0]

image.save("./magical_princess.png")
```

#### Prompt and settings for Helen Mirren:
**classic disney style helen mirren as a queen**
_Steps: 30, Sampler: Euler a, CFG scale: 7, Seed: 3496225274, Size: 512x704_

#### Prompt and settings for the Ford Model T:
**classic disney style Ford Model T - Negative prompt: person**
_Steps: 20, Sampler: DPM2 Karras, CFG scale: 7, Seed: 4817981, Size: 704x512_

This model was trained using the diffusers based dreambooth training by ShivamShrirao using prior-preservation loss and the _train-text-encoder_ flag in 9.000 steps.

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/README.md) (3.0 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/feature_extractor/preprocessor_config.json) (407.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/model_index.json) (616.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/safety_checker/model_config.json) (368.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/scheduler/scheduler_config.json) (318.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/text_encoder/model_config.json) (267.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/tokenizer/tokenizer_config.json) (776.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/unet/config.json) (1.0 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/vae/config.json) (682.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/nitrosocke/classic-anim-diffusion/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)