
# clap-htsat-fused
---


## README([From Huggingface](https://huggingface.co/laion/clap-htsat-fused))


# Model card for CLAP

Model card for CLAP: Contrastive Language-Audio Pretraining

![![clap_image](https://s3.amazonaws.com/moonup/production/uploads/1678811100805-62441d1d9fdefb55a0b7d12c.png)


#  Table of Contents

0. [TL;DR](#TL;DR)
1. [Model Details](#model-details)
2. [Usage](#usage)
3. [Uses](#uses)
4. [Citation](#citation)

# TL;DR

The abstract of the paper states that: 

> Contrastive learning has shown remarkable success in the field of multimodal representation learning. In this paper, we propose a pipeline of contrastive language-audio pretraining to develop an audio representation by combining audio data with natural language descriptions. To accomplish this target, we first release LAION-Audio-630K, a large collection of 633,526 audio-text pairs from different data sources. Second, we construct a contrastive language-audio pretraining model by considering different audio encoders and text encoders. We incorporate the feature fusion mechanism and keyword-to-caption augmentation into the model design to further enable the model to process audio inputs of variable lengths and enhance the performance. Third, we perform comprehensive experiments to evaluate our model across three tasks: text-to-audio retrieval, zero-shot audio classification, and supervised audio classification. The results demonstrate that our model achieves superior performance in text-to-audio retrieval task. In audio classification tasks, the model achieves state-of-the-art performance in the zero-shot setting and is able to obtain performance comparable to models' results in the non-zero-shot setting. LAION-Audio-630K and the proposed model are both available to the public.


# Usage

You can use this model for zero shot audio classification or extracting audio and/or textual features.

# Uses

## Perform zero-shot audio classification

### Using `pipeline`

```python
from datasets import load_dataset
from paddlenlp.transformers import pipeline

dataset = load_dataset("ashraq/esc50")
audio = dataset["train"]["audio"][-1]["array"]

audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-fused")
output = audio_classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
print(output)
>>> [{"score": 0.999, "label": "Sound of a dog"}, {"score": 0.001, "label": "Sound of vaccum cleaner"}]
```

## Run the model:

You can also get the audio and text embeddings using `ClapModel`

### Run the model on CPU:

```python
from datasets import load_dataset
from paddlenlp.transformers import ClapModel, ClapProcessor

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = librispeech_dummy[0]

model = ClapModel.from_pretrained("laion/clap-htsat-fused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pd")
audio_embed = model.get_audio_features(**inputs)
```

### Run the model on GPU:

```python
from datasets import load_dataset
from paddlenlp.transformers import ClapModel, ClapProcessor

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = librispeech_dummy[0]

model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(0)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pd").to(0)
audio_embed = model.get_audio_features(**inputs)
```


# Citation

If you are using this model for your work, please consider citing the original paper:
```
@misc{https://doi.org/10.48550/arxiv.2211.06687,
  doi = {10.48550/ARXIV.2211.06687},
  
  url = {https://arxiv.org/abs/2211.06687},
  
  author = {Wu, Yusong and Chen, Ke and Zhang, Tianyu and Hui, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  
  keywords = {Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  
  title = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/README.md) (4.3 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/config.json) (5.4 KB)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/merges.txt) (445.7 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/model_state.pdparams) (586.0 MB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/preprocessor_config.json) (537.0 B)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/special_tokens_map.json) (280.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/tokenizer.json) (2.0 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/tokenizer_config.json) (389.0 B)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/laion/clap-htsat-fused/vocab.json) (779.6 KB)


[Back to Main](../../)