
# bit-50
---


## README([From Huggingface](https://huggingface.co/google/bit-50))



# Big Transfer (BiT)

The BiT model was proposed in [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370) by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
BiT is a simple recipe for scaling up pre-training of [ResNet](resnet)-like architectures (specifically, ResNetv2). The method results in significant improvements for transfer learning.


Disclaimer: The team releasing ResNet did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

The abstract from the paper is the following:

*Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By combining a few carefully selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets. BiT performs well across a surprisingly wide range of data regimes -- from 1 example per class to 1M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis of the main components that lead to high transfer performance.*


## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=bit) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from paddlenlp.transformers import BitImageProcessor, BitForImageClassification
import paddle
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = BitImageProcessor.from_pretrained("google/bit-50")
model = BitForImageClassification.from_pretrained("google/bit-50")

inputs = feature_extractor(image, return_tensors="pd")

with paddle.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label
>>> tabby, tabby cat
```

For more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/bit).

### BibTeX entry and citation info

```bibtex
@misc{https://doi.org/10.48550/arxiv.1912.11370,
  doi = {10.48550/ARXIV.1912.11370},
  
  url = {https://arxiv.org/abs/1912.11370},
  
  author = {Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Puigcerver, Joan and Yung, Jessica and Gelly, Sylvain and Houlsby, Neil},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Big Transfer (BiT): General Visual Representation Learning},
  
  publisher = {arXiv},
  
  year = {2019},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/google/bit-50/README.md) (3.4 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/google/bit-50/config.json) (68.3 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/google/bit-50/model_state.pdparams) (97.5 MB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/google/bit-50/preprocessor_config.json) (424.0 B)


[Back to Main](../../)