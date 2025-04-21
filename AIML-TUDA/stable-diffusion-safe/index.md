
# stable-diffusion-safe
---


## README([From Huggingface](https://huggingface.co/AIML-TUDA/stable-diffusion-safe))



# Safe Stable Diffusion Model Card

Safe Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.
Safe Stable Diffusion is driven by the goal of suppressing _inappropriate_ images other large Diffusion models generate, often unexpectedly. 


Safe Stable Diffusion shares weights with the Stable Diffusion v1.5. For more information please have a look at [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [🤗's Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion).

Currently, you can use this with the [ml-research GitHub repository](https://github.com/ml-research/safe-latent-diffusion) and soon it will be available with the [🧨Diffusers library](https://github.com/huggingface/diffusers). Check [this Pull Request](https://github.com/huggingface/diffusers/pull/1244) for updates.


### Original GitHub Repository

1. Download this model with either one of these checkpoints: 
   - [v1-5-pruned-emaonly.ckpt](https://huggingface.co/AIML-TUDA/stable-diffusion-safe/resolve/main/v1-5-pruned-emaonly.ckpt) - 4.27GB, ema-only weight. uses less VRAM - suitable for inference
   - [v1-5-pruned.ckpt](https://huggingface.co/AIML-TUDA/stable-diffusion-safe/resolve/main/v1-5-pruned.ckpt) - 7.7GB, ema+non-ema weights. uses more VRAM - suitable for fine-tuning

2. Install the Safe Latent Diffusion Libray.
    ```cmd
   $ pip install git+https://github.com/ml-research/safe-latent-diffusion.git 
    ```
3. And load the SLD pipline as follows.
    ```python
    from sld import SLDPipeline
    device='cuda'
    #####################
    # Path to your local clone of the weights
    ckpt_path = ''
    ####################
    pipe = SLDPipeline.from_pretrained(
        ckpt_path,
    ).to(device)    
    ```

### Definition of <i>inappropriate</i> content

What is considered inappropriate imagery may differ based on context, setting, cultural and social predisposition as well as individual factors and is overall highly subjective. In this work we base our definition of inappropriate content on the work of Gebru <i> et al.</i>: 

> [data that] if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety.  
>
> --<cite>Gebru, Timnit, et al. "Datasheets for datasets," (2021)</cite>

Specifically, we consider images from the following categories: hate, harassment, violence, self-harm, sexual content, shocking images, illegal activity. Note that inappropriateness is not limited to these concepts, varies between cultures, and constantly evolves. 


## Model Details
- **Developed by:** Patrick Schramowski, Manuel Brack
- **Model type:** Diffusion-based text-to-image generation model with suppression of inappropriate content
- **Language(s):** English
- **License:** [The CreativeML OpenRAIL M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license) is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which our license is based.
- **Model Description:** This is a model that can be used to generate and modify images based on text prompts. It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses a fixed, pretrained text encoder ([CLIP ViT-L/14](https://arxiv.org/abs/2103.00020)) as suggested in the [Imagen paper](https://arxiv.org/abs/2205.11487). This model actively suppresses the generation of inappropriate content.
- **Resources for more information:** [GitHub Repository](https://github.com/ml-research/safe-latent-diffusion), [Paper](https://arxiv.org/abs/2211.05105).
- **Cite as:**

      @article{schramowski2022safe,
      title={Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models}, 
      author={Patrick Schramowski and Manuel Brack and Björn Deiseroth and Kristian Kersting},
      year={2022},
      journal={arXiv preprint arXiv:2211.05105}
      }

# Uses
We distinguish between the following.

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
_Note: This section is taken from the [DALLE-MINI model card](https://huggingface.co/dalle-mini/dalle-mini), but applies in the same way to Safe Stable Diffusion.


The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

#### Out-of-Scope Use
The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.
Furthermore, the default safety concept only reduces the amount of content related to the aforementioned categories: hate, harassment, violence, self-harm, sexual content, shocking images, illegal activity.
Mitigating otherwise inappropriate material is out of scope for this model and requires changes to the safety concept. 
Furthermore, the approach relies on the representations of those concepts learned by the model to identify and correct corresponding content. Cases were the model itself fails to correlate an image with our textual definition of inappropriateness are out of scope. 

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
Additionally, the models' notion of inappropriateness and definitions of the included concepts are equally biased towards their perception in western, english

### Safety Module

The intended use of this model is with the [Safety Checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) in Diffusers. 
This checker works by checking model outputs against known hard-coded NSFW concepts.
The concepts are intentionally hidden to reduce the likelihood of reverse-engineering this filter.
Specifically, the checker compares the class probability of harmful concepts in the embedding space of the `CLIPTextModel` *after generation* of the images. 
The concepts are passed into the model with the generated image and compared to a hand-engineered weight for each NSFW concept.


## Training

**Training Data**
The model developers used the following dataset for training the model:

- LAION-2B (en) and subsets thereof (see next section)

**Training Procedure**
For the training procedure of the underlying Stable Diffusion v1.5 check its [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5).

Safe Stable diffusion does not perform any additional training and only performs extensive evaluation of the Stable Diffusion v1.4 model. 

- **Hardware:** A100 GPUs

## Evaluation Results 

Evaluation of inappropriateness mitigation using the [I2P Benchmark](https://huggingface.co/datasets/AIML-TUDA/i2p) and the Stable Diffusion v1.4 checkpoint. 

| Category         | Stable Diffusion | SLD-Weak | SLD-Medium | SLD-Strong | SLD-Max |
|------------------|------------------|----------|------------|------------|---------|
| Hate             | 0.40             | 0.27     | 0.20       | 0.15       | 0.09    |
| Harassment       | 0.34             | 0.24     | 0.17       | 0.13       | 0.09    |
| Violence         | 0.43             | 0.36     | 0.23       | 0.17       | 0.14    |
| Self-harm        | 0.40             | 0.27     | 0.16       | 0.10       | 0.07    |
| Sexual           | 0.35             | 0.23     | 0.14       | 0.09       | 0.06    |
| Shocking         | 0.52             | 0.41     | 0.30       | 0.20       | 0.13    |
| Illegal activity | 0.34             | 0.23     | 0.14       | 0.09       | 0.06    |
| **Overall**      | 0.39             | 0.29     | 0.19       | 0.13       | 0.09    |


Table shows probabilities of generating inappropriate content on I2P prompts. 
SLD-Configurations refer to the following hyper-parameters:


| Config     | sld_warmup_steps | sld_guidance_scale | sld_threshold | sld_momentum_scale | sld_mom_beta |
|------------|------------------|--------------------|---------------|--------------------|--------------|
| Hyp-Weak   | 15               | 200                | 0.0           | 0.0                | -            |
| Hyp-Medium | 10               | 1000               | 0.01          | 0.3                | 0.4          |
| Hyp-Strong | 7                | 2000               | 0.025         | 0.5                | 0.7          |
| Hyp-Max    | 0                | 5000               | 1.0           | 0.5                | 0.7          |


Evaluated using 10 images for each I2G prompt.
## Environmental Impact

**Safe Stable Diffusion** **Estimated Emissions**
For evaluation and development of our approach we estimate the following CO2 emissions using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). The hardware, runtime, and compute region were utilized to estimate the carbon impact.


- **Hardware Type:** A100 PCIe 40GB & A100 SXM4 80GB
- **Hours used:** 130
- **Cloud Provider:** Private Infrastructure
- **Compute Region:** Germany
- **Carbon Emitted (Power consumption x Time x Carbon produced based on location of power grid):** 20.62 kg CO2 eq.


## Citation

```bibtex
@article{schramowski2022safe,
      title={Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models}, 
      author={Patrick Schramowski and Manuel Brack and Björn Deiseroth and Kristian Kersting},
      year={2022},
      journal={arXiv preprint arXiv:2211.05105}
}
```

*This model card was written by: Manuel Brack and is based on the [Stable Diffusion v1.5 model card](https://huggingface.co/runwayml/stable-diffusion-v1-5).*



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/README.md) (13.9 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/feature_extractor/preprocessor_config.json) (342.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/model_index.json) (601.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/safety_checker/model_config.json) (372.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/scheduler/scheduler_config.json) (316.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/text_encoder/model_config.json) (267.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/tokenizer/added_tokens.json) (2.0 B)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/tokenizer/tokenizer_config.json) (834.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/unet/config.json) (873.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/vae/config.json) (549.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/AIML-TUDA/stable-diffusion-safe/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)