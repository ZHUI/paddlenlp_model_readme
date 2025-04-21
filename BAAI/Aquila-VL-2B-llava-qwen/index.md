
# Aquila-VL-2B-llava-qwen
---


## README([From Huggingface](https://huggingface.co/BAAI/Aquila-VL-2B-llava-qwen))




# Introduction

The **Aquila-VL-2B** model is a vision-language model (VLM) trained based on the [LLava-one-vision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) framework. The [Qwen2.5-1.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model is chose as the LLM, while [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) is utilized as the vision tower.

The model was trained on our self-built Infinity-MM dataset, which contains approximately 40 million image-text pairs. This dataset is a combination of open-source data collected from the internet and synthetic instruction data generated using open-source VLM models.


We have open-sourced [Infinity-MM](https://huggingface.co/datasets/BAAI/Infinity-MM) dataset and related resources. We hope you enjoy using them!

## News 
- `2024/10/25`:  The [Aquila-VL-2B](https://huggingface.co/BAAI/Aquila-VL-2B-llava-qwen) model and [Infinity-MM](https://huggingface.co/datasets/BAAI/Infinity-MM) dataset are now available.  We have also released the [technical report](https://arxiv.org/abs/2410.18558) simultaneously.

# Evaluation

We evaluated the model using the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) tool. Whenever possible, we prioritized using the OpenAI API for test sets that support API-based evaluation.

| Benchmark                    | MiniCPM-V-2 | InternVL2-2B | XinYuan-VL-2B | Qwen2-VL-2B-Instruct | Aquila-VL-2B |
| :--------------------------- | :---------: | :----------: | :-----------: | :------------------: | :----------: |
| MMBench-EN<sub>test</sub>    |    69.4     |     73.4     |   **78.9**    |         74.9         |     78.8     |
| MMBench-CN<sub>test</sub>    |    65.9     |     70.9     |     76.1      |         73.9         |   **76.4**   |
| MMBench_V1.1<sub>test</sub>  |    65.2     |     69.7     |   **75.4**    |         72.7         |     75.2     |
| MMT-Bench<sub>test</sub>     |    54.5     |     53.3     |     57.2      |         54.8         |   **58.2**   |
| RealWorldQA                  |    55.4     |     57.3     |     63.9      |         62.6         |   **63.9**   |
| HallusionBench               |    36.8     |     38.1     |     36.0      |         41.5         |   **43.0**   |
| SEEDBench2<sub>plus</sub>    |    51.8     |     60.0     |     63.0      |         62.4         |   **63.0**   |
| LLaVABench                   |    66.1     |     64.8     |     42.4      |         52.5         |   **68.4**   |
| MMStar                       |    41.6     |     50.2     |     51.9      |         47.8         |   **54.9**   |
| POPE                         |    86.6     |     85.3     |   **89.4**    |         88.0         |     83.6     |
| MMVet                        |    44.0     |     41.1     |     42.7      |       **50.7**       |     44.3     |
| MMMU<sub>val</sub>           |    39.6     |     34.9     |     43.6      |         41.7         |   **47.4**   |
| ScienceQA<sub>test</sub>     |    80.4     |     94.1     |     86.6      |         78.1         |   **95.2**   |
| AI2D<sub>test</sub>          |    64.8     |     74.4     |     74.2      |         74.6         |   **75.0**   |
| MathVista<sub>testmini</sub> |    39.0     |     45.0     |     47.1      |         47.9         |   **59.0**   |
| MathVerse<sub>testmini</sub> |    19.8     |     24.7     |     22.2      |         21.0         |   **26.2**   |
| MathVision                   |    15.4     |     12.6     |     16.3      |         17.5         |   **18.4**   |
| DocVQA<sub>test</sub>        |    71.0     |     86.9     |     87.6      |       **89.9**       |     85.0     |
| InfoVQA<sub>test</sub>       |    40.0     |     59.5     |     59.1      |       **65.4**       |     58.3     |
| ChartQA<sub>test</sub>       |    59.6     |     71.4     |     57.1      |         73.5         |   **76.5**   |
| TextVQA<sub>val</sub>        |    74.3     |     73.5     |     77.6      |       **79.9**       |     76.4     |
| OCRVQA<sub>testcore</sub>    |    54.4     |     40.2     |     67.6      |       **68.7**       |     64.0     |
| VCR<sub>en easy</sub>        |    27.6     |     51.6     |     67.7      |         68.3         |   **70.0**   |
| OCRBench                     |     613     |     784      |      782      |       **810**        |     772      |
| Average                      |    53.5     |     58.8     |     60.9      |         62.1         |   **64.1**   |



For comparison models, evaluations were conducted in a local environment, so the scores may differ slightly from those reported in papers or on the official VLMEvalKit leaderboard.

# How to use

```python
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import requests
import copy
import torch
import warnings

warnings.filterwarnings("ignore")

pretrained = "BAAI/Aquila-VL-2B-llava-qwen"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

# load image from url
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# load image from local environment
# url = "./local_image.jpg"
# image = Image.open(url)

image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=paddle.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
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

print(text_outputs)
```



# Future Plan

* We plan to train models of various sizes.
* Future training will incorporate multi-image and video data.


## **Citation**
If you find this dataset useful, please cite the following work
```
@misc{gu2024infinitymmscalingmultimodalperformance,
      title={Infinity-MM: Scaling Multimodal Performance with Large-Scale and High-Quality Instruction Data}, 
      author={Shuhao Gu and Jialing Zhang and Siyuan Zhou and Kevin Yu and Zhaohu Xing and Liangdong Wang and Zhou Cao and Jintao Jia and Zhuoyi Zhang and Yixuan Wang and Zhenchong Hu and Bo-Wen Zhang and Jijie Li and Dong Liang and Yingli Zhao and Yulong Ao and Yaoqi Liu and Fangxiang Feng and Guang Liu},
      year={2024},
      eprint={2410.18558},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.18558}, 
}
```




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/README.md) (7.5 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/added_tokens.json) (605.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/config.json) (2.7 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/generation_config.json) (207.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/merges.txt) (1.6 MB)

- [model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/model.safetensors) (4.1 GB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/special_tokens_map.json) (613.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/tokenizer_config.json) (7.2 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila-VL-2B-llava-qwen/vocab.json) (2.6 MB)


[Back to Main](../../)