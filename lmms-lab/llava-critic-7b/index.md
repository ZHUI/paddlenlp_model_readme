
# llava-critic-7b
---


## README([From Huggingface](https://huggingface.co/lmms-lab/llava-critic-7b))



# LLaVA-Critic-7B

## Model Summary

`llava-critic-7b` is the first open-source large multimodal model (LMM) designed as a generalist evaluator for assessing model performance across diverse multimodal scenarios. Built on the foundation of `llava-onevision-7b-ov`, it has been finetuned on [LLaVA-Critic-113k](https://huggingface.co/datasets/lmms-lab/llava-critic-113k) dataset to develop its "critic" capacities.

LLaVA-Critic excels in two primary scenarios:
- 1️⃣ LMM-as-a-Judge: It delivers judgments closely aligned with human, and provides concrete, image-grounded reasons. An open-source alternative to GPT for evaluations.
- 2️⃣ Preference Learning: Reliable reward signals power up visual chat, leading to LLaVA-OV-Chat [7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov-chat)/[72B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov-chat).

For further details, please refer to the following resources:
- 📰 Paper: https://arxiv.org/abs/2410.02712
- 🪐 Project Page: https://llava-vl.github.io/blog/2024-10-03-llava-critic/
- 📦 Datasets: https://huggingface.co/datasets/lmms-lab/llava-critic-113k
- 🤗 Model Collections: https://huggingface.co/collections/lmms-lab/llava-critic-66fe3ef8c6e586d8435b4af8
- 👋 Point of Contact: [Tianyi Xiong](https://tyxiong23.github.io/)


## Use

### Intended Use

The model demonstrates general capacities in providing quantitative judgments and qualitative justifications for evaluating LMM-generated responses. It mainly focuses on two evaluation settings:
- *Pointwise scoring*, where it assigns a score to an individual candidate response.
- *Pairwise ranking*, where it compares two candidate responses to determine their relative quality.

### Quick Start

~~~python
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
import os


warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-critic-7b"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

url = "https://github.com/LLaVA-VL/blog/blob/main/2024-10-03-llava-critic/static/images/critic_img_seven.png?raw=True"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=paddle.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

# pairwise ranking
critic_prompt = "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of the answers provided by a Large Multimodal Model (LMM). Determine which answer is better and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [What this image presents?]\nThe first response: [The image is a black and white sketch of a line that appears to be in the shape of a cross. The line is a simple and straightforward representation of the cross shape, with two straight lines intersecting at a point.]\nThe second response: [This is a handwritten number seven.]\nASSISTANT:\n"

# pointwise scoring
# critic_prompt = "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of answer answers provided by a Large Multimodal Model (LMM). Score the response out of 100 and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [What this image presents?]\nThe LMM response: [This is a handwritten number seven.]\nASSISTANT:\n "

question = DEFAULT_IMAGE_TOKEN + "\n" + critic_prompt
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pd").unsqueeze(0)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
~~~


## Citation

```
@article{xiong2024llavacritic,
  title={LLaVA-Critic: Learning to Evaluate Multimodal Models},
  author={Xiong, Tianyi and Wang, Xiyao and Guo, Dong and Ye, Qinghao and Fan, Haoqi and Gu, Quanquan and Huang, Heng and Li, Chunyuan},
  year={2024},
  eprint={2410.02712},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2410.02712},
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/README.md) (5.2 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/added_tokens.json) (101.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/config.json) (3.3 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/generation_config.json) (208.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/merges.txt) (1.6 MB)

- [model-00001-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/model-00001-of-00004.safetensors) (4.5 GB)

- [model-00002-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/model-00002-of-00004.safetensors) (4.6 GB)

- [model-00003-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/model-00003-of-00004.safetensors) (4.7 GB)

- [model-00004-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/model-00004-of-00004.safetensors) (1.2 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/model.safetensors.index.json) (78.9 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/special_tokens_map.json) (367.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/tokenizer_config.json) (1.5 KB)

- [trainer_state.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/trainer_state.json) (550.1 KB)

- [training_args.bin](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/training_args.bin) (7.7 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-critic-7b/vocab.json) (2.6 MB)


[Back to Main](../../)