
# shap-e
---


## README([From Huggingface](https://huggingface.co/openai/shap-e))



# Shap-E

Shap-E introduces a diffusion process that can generate a 3D image from a text prompt. It was introduced in [Shap-E: Generating Conditional 3D Implicit Functions](https://arxiv.org/abs/2305.02463) by Heewoo Jun and Alex Nichol from OpenAI. 

Original repository of Shap-E can be found here: https://github.com/openai/shap-e. 

_The authors of Shap-E didn't author this model card. They provide a separate model card [here](https://github.com/openai/shap-e/blob/main/model-card.md)._

## Introduction 

The abstract of the Shap-E paper:

*We present Shap-E, a conditional generative model for 3D assets. Unlike recent work on 3D generative models which produce a single output representation, Shap-E directly generates the parameters of implicit functions that can be rendered as both textured meshes and neural radiance fields. We train Shap-E in two stages: first, we train an encoder that deterministically maps 3D assets into the parameters of an implicit function; second, we train a conditional diffusion model on outputs of the encoder. When trained on a large dataset of paired 3D and text data, our resulting models are capable of generating complex and diverse 3D assets in a matter of seconds. When compared to Point-E, an explicit generative model over point clouds, Shap-E converges faster and reaches comparable or better sample quality despite modeling a higher-dimensional, multi-representation output space. We release model weights, inference code, and samples at [this https URL](https://github.com/openai/shap-e).*

## Released checkpoints

The authors released the following checkpoints:

* [openai/shap-e](https://hf.co/openai/shap-e): produces a 3D image from a text input prompt
* [openai/shap-e-img2img](https://hf.co/openai/shap-e-img2img): samples a 3D image from synthetic 2D image

## Usage examples in 🧨 diffusers

First make sure you have installed all the dependencies:

```bash 
pip install transformers accelerate -q
pip install git+https://github.com/huggingface/diffusers@@shap-ee
```

Once the dependencies are installed, use the code below:

```python 
import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif


ckpt_id = "openai/shap-e"
pipe = ShapEPipeline.from_pretrained(repo).to("cuda")


guidance_scale = 15.0
prompt = "a shark"
images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    size=256,
).images

gif_path = export_to_gif(images, "shark_3d.gif")
```

## Results 

<table>
    <tbody>
        <tr>
            <td align="center">
                <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/shap-e/bird_3d.gif" alt="a bird">
            </td>
            <td align="center">
                <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/shap-e/shark_3d.gif" alt="a shark">
            </td align="center">
            <td align="center">
                <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/shap-e/veg_3d.gif" alt="A bowl of vegetables">
            </td>
        </tr>
        <tr>
            <td align="center">A bird</td>
            <td align="center">A shark</td>
            <td align="center">A bowl of vegetables</td>
        </tr>
     </tr> 
    </tbody>
<table>

## Training details

Refer to the [original paper](https://arxiv.org/abs/2305.02463).

## Known limitations and potential biases
    
Refer to the [original model card](https://github.com/openai/shap-e/blob/main/model-card.md).
    
## Citation

```bibtex 
@misc{jun2023shape,
      title={Shap-E: Generating Conditional 3D Implicit Functions}, 
      author={Heewoo Jun and Alex Nichol},
      year={2023},
      eprint={2305.02463},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/README.md) (3.8 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/model_index.json) (425.0 B)

- [prior/config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/prior/config.json) (617.0 B)

- [prior/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/prior/model_state.pdparams) (1.2 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/scheduler/scheduler_config.json) (375.0 B)

- [shap_e_renderer/config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/shap_e_renderer/config.json) (584.0 B)

- [shap_e_renderer/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/shap_e_renderer/diffusion_paddle_model.safetensors) (863.3 MB)

- [shap_e_renderer/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/shap_e_renderer/model_state.pdparams) (863.3 MB)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/text_encoder/config.json) (650.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/text_encoder/model_state.pdparams) (471.7 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/tokenizer/tokenizer_config.json) (812.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/openai/shap-e/tokenizer/vocab.json) (1.0 MB)


[Back to Main](../../)