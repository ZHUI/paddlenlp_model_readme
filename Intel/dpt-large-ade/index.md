
# dpt-large-ade
---


## README([From Huggingface](https://huggingface.co/Intel/dpt-large-ade))



# DPT (large-sized model) fine-tuned on ADE20k

The model is used for semantic segmentation of input images such as seen in the table below:


| Input Image | Output Segmented Image | 
| --- | --- | 
| ![![input image](https://cdn-uploads.huggingface.co/production/uploads/641bd18baebaa27e0753f2c9/cG0alacJ4MeSL18CneD2u.png) | ![![Segmented image](https://cdn-uploads.huggingface.co/production/uploads/641bd18baebaa27e0753f2c9/G3g6Bsuti60-bCYzgbt5o.png)| 

## Model description

The Midas 3.0 nbased Dense Prediction Transformer (DPT) model was trained on ADE20k for semantic segmentation. It was introduced in the paper 
[Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413) by Ranftl et al. and first released in [this repository](https://github.com/isl-org/DPT). 


The MiDaS v3.0 DPT uses the Vision Transformer (ViT) as backbone and adds a neck + head on top for semantic segmentation.

![![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dpt_architecture.jpg)

Disclaimer: The team releasing DPT did not write a model card for this model so this model card has been written by the Hugging Face and the Intel AI Community team.


## Results:
According to the authors, at the time of publication, when applied to semantic segmentation, dense vision transformers set a new state of the art on 

**ADE20K with 49.02% mIoU.**

We further show that the architecture can be fine-tuned on smaller datasets such as NYUv2, KITTI, and Pascal Context where it also sets the new state of the art. Our models are available at
[Intel DPT GItHub Repository](https://github.com/intel-isl/DPT).

## Intended uses & limitations

You can use the raw model for semantic segmentation. See the [model hub](https://huggingface.co/models?search=dpt) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model:

```python
from paddlenlp.transformers import DPTFeatureExtractor, DPTForSemanticSegmentation
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000026204.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = DPTImageProcessor .from_pretrained("Intel/dpt-large-ade")
model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits
print(logits.shape)
logits
prediction = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # Reverse the size of the original image (width, height)
    mode="bicubic",
    align_corners=False
)

# Convert logits to class predictions
prediction = torch.argmax(prediction, dim=1) + 1

# Squeeze the prediction tensor to remove dimensions
prediction = prediction.squeeze()

# Move the prediction tensor to the CPU and convert it to a numpy array
prediction = prediction.cpu().numpy()

# Convert the prediction array to an image
predicted_seg = Image.fromarray(prediction.squeeze().astype('uint8'))

# Define the ADE20K palette
adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]

# Apply the color map to the predicted segmentation image
predicted_seg.putpalette(adepallete)

# Blend the original image and the predicted segmentation image
out = Image.blend(image, predicted_seg.convert("RGB"), alpha=0.5)

out
```

For more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/master/en/model_doc/dpt).

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2103-13413,
  author    = {Ren{\'{e}} Ranftl and
               Alexey Bochkovskiy and
               Vladlen Koltun},
  title     = {Vision Transformers for Dense Prediction},
  journal   = {CoRR},
  volume    = {abs/2103.13413},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.13413},
  eprinttype = {arXiv},
  eprint    = {2103.13413},
  timestamp = {Wed, 07 Apr 2021 15:31:46 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-13413.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Intel/dpt-large-ade/README.md) (5.8 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Intel/dpt-large-ade/config.json) (7.0 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Intel/dpt-large-ade/model_state.pdparams) (1.3 GB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Intel/dpt-large-ade/preprocessor_config.json) (382.0 B)


[Back to Main](../../)