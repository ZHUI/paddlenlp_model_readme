
# clip-vit-base-patch32
---


## README([From Huggingface](https://huggingface.co/openai/clip-vit-base-patch32))

---
tags:
- vision
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/cat-dog-music.png
  candidate_labels: playing music, playing sports
  example_title: Cat & Dog
---

# Model Card: CLIP

Disclaimer: The model card is taken and modified from the official CLIP repository, it can be found [here](https://github.com/openai/CLIP/blob/main/model-card.md).

## Model Details

The CLIP model was developed by researchers at OpenAI to learn about what contributes to robustness in computer vision tasks. The model was also developed to test the ability of models to generalize to arbitrary image classification tasks in a zero-shot manner. It was not developed for general model deployment - to deploy models like CLIP, researchers will first need to carefully study their capabilities in relation to the specific context they’re being deployed within.

### Model Date

January 2021

### Model Type

The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss. 

The original implementation had two variants: one using a ResNet image encoder and the other using a Vision Transformer. This repository has the variant with the Vision Transformer.


### Documents

- [Blog Post](https://openai.com/blog/clip/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)


### Use with Transformers

```python3
from PIL import Image
import requests

from paddlenlp.transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pd", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
```


## Model Use

### Intended Use

The model is intended as a research output for research communities. We hope that this model will enable researchers to better understand and explore zero-shot, arbitrary image classification. We also hope it can be used for interdisciplinary studies of the potential impact of such models - the CLIP paper includes a discussion of potential downstream impacts to provide an example for this sort of analysis.

#### Primary intended uses

The primary intended users of these models are AI researchers.

We primarily imagine the model will be used by researchers to better understand robustness, generalization, and other capabilities, biases, and constraints of computer vision models.

### Out-of-Scope Use Cases

**Any** deployed use case of the model - whether commercial or not - is currently out of scope. Non-deployed use cases such as image search in a constrained environment, are also not recommended unless there is thorough in-domain testing of the model with a specific, fixed class taxonomy. This is because our safety assessment demonstrated a high need for task specific testing especially given the variability of CLIP’s performance with different class taxonomies. This makes untested and unconstrained deployment of the model in any use case currently potentially harmful. 

Certain use cases which would fall under the domain of surveillance and facial recognition are always out-of-scope regardless of performance of the model. This is because the use of artificial intelligence for tasks such as these can be premature currently given the lack of testing norms and checks to ensure its fair use.

Since the model has not been purposefully trained in or evaluated on any languages other than English, its use should be limited to English language use cases.



## Data

The model was trained on publicly available image-caption data. This was done through a combination of crawling a handful of websites and using commonly-used pre-existing image datasets such as [YFCC100M](http://projects.dfki.uni-kl.de/yfcc100m/). A large portion of the data comes from our crawling of the internet. This means that the data is more representative of people and societies most connected to the internet which tend to skew towards more developed nations, and younger, male users.

### Data Mission Statement

Our goal with building this dataset was to test out robustness and generalizability in computer vision tasks. As a result, the focus was on gathering large quantities of data from different publicly-available internet data sources. The data was gathered in a mostly non-interventionist manner. However, we only crawled websites that had policies against excessively violent and adult images and allowed us to filter out such content. We do not intend for this dataset to be used as the basis for any commercial or deployed model and will not be releasing the dataset.



## Performance and Limitations

### Performance

We have evaluated the performance of CLIP on a wide range of benchmarks across a variety of computer vision datasets such as OCR to texture recognition to fine-grained classification. The paper describes model performance on the following datasets:

- Food101
- CIFAR10   
- CIFAR100   
- Birdsnap
- SUN397
- Stanford Cars
- FGVC Aircraft
- VOC2007
- DTD
- Oxford-IIIT Pet dataset
- Caltech101
- Flowers102
- MNIST   
- SVHN 
- IIIT5K   
- Hateful Memes   
- SST-2
- UCF101
- Kinetics700
- Country211
- CLEVR Counting
- KITTI Distance
- STL-10
- RareAct
- Flickr30
- MSCOCO
- ImageNet
- ImageNet-A
- ImageNet-R
- ImageNet Sketch
- ObjectNet (ImageNet Overlap)
- Youtube-BB
- ImageNet-Vid

## Limitations

CLIP and our analysis of it have a number of limitations. CLIP currently struggles with respect to certain tasks such as fine grained classification and counting objects. CLIP also poses issues with regards to fairness and bias which we discuss in the paper and briefly in the next section. Additionally, our approach to testing CLIP also has an important limitation- in many cases we have used linear probes to evaluate the performance of CLIP and there is evidence suggesting that linear probes can underestimate model performance.

### Bias and Fairness

We find that the performance of CLIP - and the specific biases it exhibits - can depend significantly on class design and the choices one makes for categories to include and exclude. We tested the risk of certain kinds of denigration with CLIP by classifying images of people from [Fairface](https://arxiv.org/abs/1908.04913) into crime-related and non-human animal categories. We found significant disparities with respect to race and gender. Additionally, we found that these disparities could shift based on how the classes were constructed. (Details captured in the Broader Impacts Section in the paper).

We also tested the performance of CLIP on gender, race and age classification using the Fairface dataset (We default to using race categories as they are constructed in the Fairface dataset.) in order to assess quality of performance across different demographics. We found accuracy >96% across all races for gender classification with ‘Middle Eastern’ having the highest accuracy (98.4%) and ‘White’ having the lowest (96.5%). Additionally, CLIP averaged ~93% for racial classification and ~63% for age classification. Our use of evaluations to test for gender, race and age classification as well as denigration harms is simply to evaluate performance of the model across people and surface potential risks and not to demonstrate an endorsement/enthusiasm for such tasks.



## Feedback

### Where to send questions or comments about the model

Please use [this Google Form](https://forms.gle/Uv7afRH5dvY34ZEs9)



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/README.md) (7.8 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/config.json) (4.1 KB)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/merges.txt) (512.4 KB)

- [model_config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/model_config.json) (381.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/model_state.pdparams) (577.1 MB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/preprocessor_config.json) (342.0 B)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/special_tokens_map.json) (478.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/tokenizer_config.json) (194.0 B)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/openai/clip-vit-base-patch32/vocab.json) (842.1 KB)


[Back to Main](../../)