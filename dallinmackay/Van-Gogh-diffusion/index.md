
# Van-Gogh-diffusion
---


## README([From Huggingface](https://huggingface.co/dallinmackay/Van-Gogh-diffusion))


### Van Gogh Diffusion

v2 - fixed and working

This is a fine-tuned Stable Diffusion model (based on v1.5) trained on screenshots from the film **_Loving Vincent_**. Use the token **_lvngvncnt_** at the BEGINNING of your prompts to use the style (e.g., "lvngvncnt, beautiful woman at sunset"). This model works best with the Euler sampler (NOT Euler_a).

_Download the ckpt file from "files and versions" tab into the stable diffusion models folder of your web-ui of choice._

If you get too many yellow faces or you dont like the strong blue bias, simply put them in the negative prompt (e.g., "Yellow face, blue").

--

**Characters rendered with this model:**
![![Character Samples](https://huggingface.co/dallinmackay/Van-Gogh-diffusion/resolve/main/preview1.jpg)
  _prompt and settings used: **lvngvncnt, [person], highly detailed** | **Steps: 25, Sampler: Euler, CFG scale: 6**_

--

**Landscapes/miscellaneous rendered with this model:**
![![Landscape Samples](https://huggingface.co/dallinmackay/Van-Gogh-diffusion/resolve/main/preview2.jpg)
  _prompt and settings used: **lvngvncnt, [subject/setting], highly detailed** | **Steps: 25, Sampler: Euler, CFG scale: 6**_

--

This model was trained with Dreambooth, using TheLastBen colab notebook
--
### 🧨 Diffusers

This model can be used just like any other Stable Diffusion model. For more information,
please have a look at the [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion).

You can also export the model to [ONNX](https://huggingface.co/docs/diffusers/optimization/onnx), [MPS](https://huggingface.co/docs/diffusers/optimization/mps) and/or [FLAX/JAX]().

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "dallinmackay/Van-Gogh-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "lvngvncnt, beautiful woman at sunset"
image = pipe(prompt).images[0]

image.save("./sunset.png")
```

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

--
[![![Become A Patreon](https://badgen.net/badge/become/a%20patron/F96854)](https://www.patreon.com/dallinmackay)



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/README.md) (3.1 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/feature_extractor/preprocessor_config.json) (407.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/model_index.json) (616.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/safety_checker/model_config.json) (368.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/scheduler/scheduler_config.json) (318.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/text_encoder/model_config.json) (267.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/tokenizer/tokenizer_config.json) (772.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/unet/config.json) (1.0 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/vae/config.json) (673.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/dallinmackay/Van-Gogh-diffusion/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)