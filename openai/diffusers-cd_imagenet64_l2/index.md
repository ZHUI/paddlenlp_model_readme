
# diffusers-cd_imagenet64_l2
---


## README([From Huggingface](https://huggingface.co/openai/diffusers-cd_imagenet64_l2))



**Disclaimer**: This model was added by the amazing community contributors [dg845](https://huggingface.co/dg845) and [ayushtues](https://huggingface.co/ayushtues)❤️

Consistency models are a new class of generative models introduced in ["Consistency Models"](https://arxiv.org/abs/2303.01469) ([paper](https://arxiv.org/pdf/2303.01469.pdf), [code](https://github.com/openai/consistency_models)) by Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever.
From the paper abstract:

> Diffusion models have significantly advanced the fields of image, audio, and video generation, but
they depend on an iterative sampling process that causes slow generation. To overcome this limitation,
we propose consistency models, a new family of models that generate high quality samples by directly
mapping noise to data. They support fast one-step generation by design, while still allowing multistep
sampling to trade compute for sample quality. They also support zero-shot data editing, such as image
inpainting, colorization, and super-resolution, without requiring explicit training on these tasks.
Consistency models can be trained either by distilling pre-trained diffusion models, or as standalone
generative models altogether. Through extensive experiments, we demonstrate that they outperform
existing distillation techniques for diffusion models in one- and few-step sampling, achieving the new
state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet 64 x 64 for one-step generation. When
trained in isolation, consistency models become a new family of generative models that can outperform
existing one-step, non-adversarial generative models on standard benchmarks such as CIFAR-10, ImageNet
64 x 64 and LSUN 256 x 256.

Intuitively, a consistency model can be thought of as a model which, when evaluated on a noisy image and timestep, returns an output image sample similar to that which would be returned by running a sampling algorithm on a diffusion model.
Consistency models can be parameterized by any neural network whose input has the same dimensionality as its output, such as a U-Net.

More precisely, given a teacher diffusion model and fixed sampler, we can train ("distill") a consistency model such that when it is given a noisy image and its corresponding timestep, the output sample of the consistency model will be close to the output that would result by using the sampler on the diffusion model to produce a sample, starting at the same noisy image and timestep.
The authors call this procedure "consistency distillation (CD)".
Consistency models can also be trained from scratch to generate clean images from a noisy image and timestep, which the authors call "consistency training (CT)".

This model is a `diffusers`-compatible version of the [cd_imagenet64_l2.pt](https://github.com/openai/consistency_models#pre-trained-models) checkpont from the [original code and model release](https://github.com/openai/consistency_models).
This model was distilled (via consistency distillation (CD)) from an [EDM model](https://arxiv.org/pdf/2206.00364.pdf) trained on the ImageNet 64x64 dataset, using the [L2 distance](https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) as the measure of closeness.
See the [original model card](https://github.com/openai/consistency_models/blob/main/model-card.md) for more information.

## Download

The original PyTorch model checkpoint can be downloaded from the [original code and model release](https://github.com/openai/consistency_models#pre-trained-models). 

The `diffusers` pipeline for the `cd-imagenet64-l2` model can be downloaded as follows:

```python
from diffusers import ConsistencyModelPipeline

pipe = ConsistencyModelPipeline.from_pretrained("openai/diffusers-cd_imagenet64_l2")
```

## Usage

The original model checkpoint can be used with the [original consistency models codebase](https://github.com/openai/consistency_models).

Here is an example of using the `cd-imagenet64-l2` checkpoint with `diffusers`:

```python
import torch

from diffusers import ConsistencyModelPipeline

device = "cuda"
# Load the cd_imagenet64_l2 checkpoint.
model_id_or_path = "openai/diffusers-cd_imagenet64_l2"
pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, dtype=paddle.float16)
pipe

# Onestep Sampling
image = pipe(num_inference_steps=1).images[0]
image.save("cd_imagenet64_l2_onestep_sample.png")

# Onestep sampling, class-conditional image generation
# ImageNet-64 class label 145 corresponds to king penguins
image = pipe(num_inference_steps=1, class_labels=145).images[0]
image.save("cd_imagenet64_l2_onestep_sample_penguin.png")

# Multistep sampling, class-conditional image generation
# Timesteps can be explicitly specified; the particular timesteps below are from the original Github repo:
# https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
image = pipe(num_inference_steps=None, timesteps=[22, 0], class_labels=145).images[0]
image.save("cd_imagenet64_l2_multistep_sample_penguin.png")
```

## Model Details
- **Model type:** Consistency model unconditional image generation model, distilled from a diffusion model
- **Dataset:** ImageNet 64x64
- **License:** MIT
- **Model Description:** This model performs unconditional image generation. Its main component is a U-Net, which parameterizes the consistency model. This model was distilled by the Consistency Model authors from an EDM diffusion model, also originally trained by the authors.
- **Resources for more information:**: [Paper](https://arxiv.org/abs/2303.01469), [GitHub Repository](https://github.com/openai/consistency_models), [Original Model Card](/openai/consistency_models/blob/main/model-card.md)

## Datasets

_Note: This section is taken from the ["Datasets" section of the original model card](https://github.com/openai/consistency_models/blob/main/model-card.md#datasets)_.

The models that we are making available have been trained on the [ILSVRC 2012 subset of ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) or on individual categories from [LSUN](https://arxiv.org/abs/1506.03365). Here we outline the characteristics of these datasets that influence the behavior of the models:

**ILSVRC 2012 subset of ImageNet**: This dataset was curated in 2012 and has around a million pictures, each of which belongs to one of 1,000 categories. A significant number of the categories in this dataset are animals, plants, and other naturally occurring objects. Although many photographs include humans, these humans are typically not represented by the class label (for example, the category "Tench, tinca tinca" includes many photographs of individuals holding fish).

**LSUN**: This dataset was collected in 2015 by a combination of human labeling via Amazon Mechanical Turk and automated data labeling. Both classes that we consider have more than a million images. The dataset creators discovered that when assessed by trained experts, the label accuracy was approximately 90% throughout the entire LSUN dataset. The pictures are gathered from the internet, and those in the cat class often follow a "meme" format. Occasionally, people, including faces, appear in these photographs.

## Performance

_Note: This section is taken from the ["Performance" section of the original model card](https://github.com/openai/consistency_models/blob/main/model-card.md#performance)_.

These models are intended to generate samples consistent with their training distributions.
This has been measured in terms of FID, Inception Score, Precision, and Recall.
These metrics all rely on the representations of a [pre-trained Inception-V3 model](https://arxiv.org/abs/1512.00567),
which was trained on ImageNet, and so is likely to focus more on the ImageNet classes (such as animals) than on other visual features (such as human faces).

## Intended Use

_Note: This section is taken from the ["Intended Use" section of the original model card](https://github.com/openai/consistency_models/blob/main/model-card.md#intended-use)_.

These models are intended to be used for research purposes only. In particular, they can be used as a baseline for generative modeling research, or as a starting point for advancing such research. These models are not intended to be commercially deployed. Additionally, they are not intended to be used to create propaganda or offensive imagery.

## Limitations

_Note: This section is taken from the ["Limitations" section of the original model card](https://github.com/openai/consistency_models/blob/main/model-card.md#limitations)_.

These models sometimes produce highly unrealistic outputs, particularly when generating images containing human faces.
This may stem from ImageNet's emphasis on non-human objects.

In consistency distillation and training, minimizing LPIPS results in better sample quality, as evidenced by improved FID and Inception scores. However, it also carries the risk of overestimating model performance, because LPIPS uses a VGG network pre-trained on ImageNet, while FID and Inception scores also rely on convolutional neural networks (the Inception network in particular) pre-trained on the same ImageNet dataset. Although these two convolutional neural networks do not share the same architecture and we extract latents from them in substantially different ways, knowledge leakage is still plausible which can undermine the fidelity of FID and Inception scores.

Because ImageNet and LSUN contain images from the internet, they include photos of real people, and the model may have memorized some of the information contained in these photos. However, these images are already publicly available, and existing generative models trained on ImageNet have not demonstrated significant leakage of this information.





## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/openai/diffusers-cd_imagenet64_l2/README.md) (9.7 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/openai/diffusers-cd_imagenet64_l2/model_index.json) (216.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/diffusers-cd_imagenet64_l2/scheduler/scheduler_config.json) (240.0 B)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/diffusers-cd_imagenet64_l2/unet/config.json) (1.1 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/openai/diffusers-cd_imagenet64_l2/unet/model_state.pdparams) (1.1 GB)


[Back to Main](../../)