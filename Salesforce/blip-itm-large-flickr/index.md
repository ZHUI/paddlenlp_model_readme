
# blip-itm-large-flickr
---


## README([From Huggingface](https://huggingface.co/Salesforce/blip-itm-large-flickr))

---
pipeline_tags: 'other'
tags:
  - image-text-matching
languages:
  - en
license: bsd-3-clause
---

# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

Model card for BLIP trained on image-text matching - large architecture (with ViT large backbone) trained on Flickr30k dataset.

| ![![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif) |
|:--:|
| <b> Pull figure from BLIP official repo | Image source: https://github.com/salesforce/BLIP </b>|

## TL;DR

Authors from the [paper](https://arxiv.org/abs/2201.12086) write in the abstract:

*Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released.*

## Usage

You can use this model for conditional and un-conditional image captioning

### Using the Pytorch model

#### Running the model on CPU

<details>
<summary> Click to expand </summary>

```python
import requests
from PIL import Image
from paddlenlp.transformers import BlipProcessor, BlipForImageTextRetrieval

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-flickr")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-flickr")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "A woman and a dog sitting together in a beach."
inputs = processor(raw_image, question, return_tensors="pd")

itm_scores = model(**inputs)[0]
cosine_score = model(**inputs, use_itm_head=False)[0]
```
</details>

#### Running the model on GPU

##### In full precision 

<details>
<summary> Click to expand </summary>

```python
import requests
from PIL import Image
from paddlenlp.transformers import BlipProcessor, BlipForImageTextRetrieval

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-flickr")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-flickr")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "A woman and a dog sitting together in a beach."
inputs = processor(raw_image, question, return_tensors="pd")

itm_scores = model(**inputs)[0]
cosine_score = model(**inputs, use_itm_head=False)[0]
```
</details>

##### In half precision (`float16`)

<details>
<summary> Click to expand </summary>

```python
import paddle
import requests
from PIL import Image
from paddlenlp.transformers import BlipProcessor, BlipForImageTextRetrieval

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-flickr")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-flickr", dtype=paddle.float16)

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "A woman and a dog sitting together in a beach."
inputs = processor(raw_image, question, return_tensors="pd").to("cuda", paddle.float16)

itm_scores = model(**inputs)[0]
cosine_score = model(**inputs, use_itm_head=False)[0]
```
</details>

## Ethical Considerations
This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.

## BibTex and citation info

```
@misc{https://doi.org/10.48550/arxiv.2201.12086,
  doi = {10.48550/ARXIV.2201.12086},
  
  url = {https://arxiv.org/abs/2201.12086},
  
  author = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip-itm-large-flickr/README.md) (5.4 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip-itm-large-flickr/config.json) (4.5 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip-itm-large-flickr/model_state.pdparams) (1.7 GB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip-itm-large-flickr/preprocessor_config.json) (445.0 B)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip-itm-large-flickr/special_tokens_map.json) (125.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip-itm-large-flickr/tokenizer_config.json) (456.0 B)

- [vocab.txt](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip-itm-large-flickr/vocab.txt) (226.1 KB)


[Back to Main](../../)