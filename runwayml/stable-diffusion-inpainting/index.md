
# stable-diffusion-inpainting
---


## README([From Huggingface](https://huggingface.co/runwayml/stable-diffusion-inpainting))



# Stable Diffusion Inpainting model card

### ⚠️ This repository is a mirror of the now deprecated `ruwnayml/stable-diffusion-inpainting`, this repository or oganization are not affiliated in any way with RunwayML.
Modifications to the original model card are in <span style="color:crimson">red</span> or <span style="color:darkgreen">green</span>


Stable Diffusion Inpainting is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input, with the extra capability of inpainting the pictures by using a mask.

The **Stable-Diffusion-Inpainting** was initialized with the weights of the [Stable-Diffusion-v-1-2](https://steps/huggingface.co/CompVis/stable-diffusion-v-1-2-original). First 595k steps regular training, then 440k steps of inpainting training at resolution 512x512 on “laion-aesthetics v2 5+” and 10% dropping of the text-conditioning to improve classifier-free [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598). For inpainting, the UNet has 5 additional input channels (4 for the encoded masked-image and 1 for the mask itself) whose weights were zero-initialized after restoring the non-inpainting checkpoint. During training, we generate synthetic masks and in 25% mask everything.

[Open In Spaces](https://huggingface.co/spaces/sd-legacy/stable-diffusion-inpainting) | [![![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/in_painting_with_stable_diffusion_using_diffusers.ipynb)
:-------------------------:|:-------------------------:|
## Examples:

You can use this both with the [🧨Diffusers library](https://github.com/huggingface/diffusers) and [RunwayML GitHub repository](https://github.com/runwayml/stable-diffusion) (<span style="color:crimson">now deprecated</span>), <span style="color:darkgreen">Automatic1111</span>.

### Use with Diffusers

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting",
    revision="fp16",
    dtype=paddle.float16,
)
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")
```

**How it works:**
`image`          | `mask_image`
:-------------------------:|:-------------------------:|
<img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png" alt="drawing" width="300"/> | <img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png" alt="drawing" width="300"/>


`prompt`          | `Output`
:-------------------------:|:-------------------------:|
<span style="position: relative;bottom: 150px;">Face of a yellow cat, high resolution, sitting on a park bench</span> | <img src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/test.png" alt="drawing" width="300"/>

### Use with Original GitHub Repository <span style="color:darkgreen">or AUTOMATIC1111</span>

1. Download the weights [sd-v1-5-inpainting.ckpt](https://huggingface.co/sd-legacy/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt)
2. Follow instructions [here](https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion) (<span style="color:crimson">now deprecated</span>). 
3. <span style="color:darkgreen">Use it with <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">AUTOMATIC1111</a></span>

## Model Details
- **Developed by:** Robin Rombach, Patrick Esser
- **Model type:** Diffusion-based text-to-image generation model
- **Language(s):** English
- **License:** [The CreativeML OpenRAIL M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license) is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which our license is based.
- **Model Description:** This is a model that can be used to generate and modify images based on text prompts. It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses a fixed, pretrained text encoder ([CLIP ViT-L/14](https://arxiv.org/abs/2103.00020)) as suggested in the [Imagen paper](https://arxiv.org/abs/2205.11487).
- **Resources for more information:** [GitHub Repository](https://github.com/runwayml/stable-diffusion), [Paper](https://arxiv.org/abs/2112.10752).
- **Cite as:**

      @InProceedings{Rombach_2022_CVPR,
          author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
          title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
          booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
          month     = {June},
          year      = {2022},
          pages     = {10684-10695}
      }

# Uses

## Direct Use 
The model is intended for research purposes only. Possible research areas and
tasks include

- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.

Excluded uses are described below.

 ### Misuse, Malicious Use, and Out-of-Scope Use
_Note: This section is taken from the [DALLE-MINI model card](https://huggingface.co/dalle-mini/dalle-mini), but applies in the same way to Stable Diffusion v1_.


The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.
#### Out-of-Scope Use
The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.
#### Misuse and Malicious Use
Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

## Limitations and Bias

### Limitations

- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.
- The model was trained mainly with English captions and will not work as well in other languages.
- The autoencoding part of the model is lossy
- The model was trained on a large-scale dataset
  [LAION-5B](https://laion.ai/blog/laion-5b/) which contains adult material
  and is not fit for product use without additional safety mechanisms and
  considerations.
- No additional measures were used to deduplicate the dataset. As a result, we observe some degree of memorization for images that are duplicated in the training data.
  The training data can be searched at [https://rom1504.github.io/clip-retrieval/](https://rom1504.github.io/clip-retrieval/) to possibly assist in the detection of memorized images.
  
### Bias
While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. 
Stable Diffusion v1 was trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/), 
which consists of images that are primarily limited to English descriptions. 
Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. 
This affects the overall output of the model, as white and western cultures are often set as the default. Further, the 
ability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.


## Training

**Training Data**
The model developers used the following dataset for training the model:

- LAION-2B (en) and subsets thereof (see next section)

**Training Procedure**
Stable Diffusion v1 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training, 

- Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4
- Text prompts are encoded through a ViT-L/14 text-encoder.
- The non-pooled output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.
- The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet.

We currently provide six checkpoints, `sd-v1-1.ckpt`, `sd-v1-2.ckpt` and `sd-v1-3.ckpt`, `sd-v1-4.ckpt`, `sd-v1-5.ckpt` and `sd-v1-5-inpainting.ckpt`
which were trained as follows,

- `sd-v1-1.ckpt`: 237k steps at resolution `256x256` on [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en).
  194k steps at resolution `512x512` on [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution) (170M examples from LAION-5B with resolution `>= 1024x1024`).
- `sd-v1-2.ckpt`: Resumed from `sd-v1-1.ckpt`.
  515k steps at resolution `512x512` on "laion-improved-aesthetics" (a subset of laion2B-en,
filtered to images with an original size `>= 512x512`, estimated aesthetics score `> 5.0`, and an estimated watermark probability `< 0.5`. The watermark estimate is from the LAION-5B metadata, the aesthetics score is estimated using an [improved aesthetics estimator](https://github.com/christophschuhmann/improved-aesthetic-predictor)).
- `sd-v1-3.ckpt`: Resumed from `sd-v1-2.ckpt`. 195k steps at resolution `512x512` on "laion-improved-aesthetics" and 10\% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).
- `sd-v1-4.ckpt`: Resumed from stable-diffusion-v1-2.225,000 steps at resolution 512x512 on "laion-aesthetics v2 5+" and 10 % dropping of the text-conditioning to [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).
- `sd-v1-5.ckpt`: Resumed from sd-v1-2.ckpt. 595k steps at resolution 512x512 on "laion-aesthetics v2 5+" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling.
- `sd-v1-5-inpaint.ckpt`: Resumed from sd-v1-2.ckpt. 595k steps at resolution 512x512 on "laion-aesthetics v2 5+" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling. Then 440k steps of inpainting training at resolution 512x512 on “laion-aesthetics v2 5+” and 10% dropping of the text-conditioning. For inpainting, the UNet has 5 additional input channels (4 for the encoded masked-image and 1 for the mask itself) whose weights were zero-initialized after restoring the non-inpainting checkpoint. During training, we generate synthetic masks and in 25% mask everything.


- **Hardware:** 32 x 8 x A100 GPUs
- **Optimizer:** AdamW
- **Gradient Accumulations**: 2
- **Batch:** 32 x 8 x 2 x 4 = 2048
- **Learning rate:** warmup to 0.0001 for 10,000 steps and then kept constant

## Evaluation Results 
Evaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0,
5.0, 6.0, 7.0, 8.0) and 50 PLMS sampling
steps show the relative improvements of the checkpoints:

![![pareto](https://huggingface.co/CompVis/stable-diffusion/resolve/main/v1-1-to-v1-5.png) 

Evaluated using 50 PLMS steps and 10000 random prompts from the COCO2017 validation set, evaluated at 512x512 resolution.  Not optimized for FID scores.

## Inpainting Evaluation
To assess the performance of the inpainting model, we used the same evaluation
protocol as in our [LDM paper](https://arxiv.org/abs/2112.10752). Since the
Stable Diffusion Inpainting Model acccepts a text input, we simply used a fixed
prompt of `photograph of a beautiful empty scene, highest quality settings`.

| Model                       | FID  | LPIPS            |
|-----------------------------|------|------------------|
| Stable Diffusion Inpainting | 1.00 | 0.141 (+- 0.082) |
| Latent Diffusion Inpainting | 1.50 | 0.137 (+- 0.080) |
| CoModGAN                    | 1.82 | 0.15             |
| LaMa                        | 2.21 | 0.134 (+- 0.080) |

## Environmental Impact

**Stable Diffusion v1** **Estimated Emissions**
Based on that information, we estimate the following CO2 emissions using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). The hardware, runtime, cloud provider, and compute region were utilized to estimate the carbon impact.

- **Hardware Type:** A100 PCIe 40GB
- **Hours used:** 150000
- **Cloud Provider:** AWS
- **Compute Region:** US-east
- **Carbon Emitted (Power consumption x Time x Carbon produced based on location of power grid):** 11250 kg CO2 eq.


## Citation

```bibtex
    @InProceedings{Rombach_2022_CVPR,
        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
        title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {10684-10695}
    }
```

*This model card was written by: Robin Rombach and Patrick Esser and is based on the [DALL-E Mini model card](https://huggingface.co/dalle-mini/dalle-mini).*



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/README.md) (14.8 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/feature_extractor/preprocessor_config.json) (342.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/model_index.json) (601.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/safety_checker/model_config.json) (614.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/scheduler/scheduler_config.json) (309.0 B)

- [scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/scheduler_config.json) (309.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/text_encoder/config.json) (592.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/text_encoder/model_config.json) (463.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/tokenizer/added_tokens.json) (2.0 B)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/tokenizer/merges.txt) (512.4 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/tokenizer/special_tokens_map.json) (478.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/tokenizer/tokenizer_config.json) (312.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/tokenizer/vocab.json) (842.1 KB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/unet/config.json) (749.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/vae/config.json) (558.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/runwayml/stable-diffusion-inpainting/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)