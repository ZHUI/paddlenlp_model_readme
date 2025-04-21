
# InternVL2-8B-MPO
---


## README([From Huggingface](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO))


# InternVL2-8B-MPO

[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl2.0_mpo)  [\[🆕 Blog\]](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/)  [\[📜 Paper\]](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/) [\[📖 Documents\]](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html)

[切换至中文版](#简介)

![![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/619507e7b74b6c591f794340/sy8aVC1Y5wtAjG-OQzrDI.jpeg)

## Introduction

Existing open-source multimodal large language models (MLLMs) generally follow a training process involving pre-training and supervised fine-tuning. However, these models suffer from distribution shifts, which limit their multimodal reasoning, particularly in the Chain-of-Thought (CoT) performance.

To address this, we introduce a preference optimization (PO) process to enhance the multimodal reasoning capabilities of MLLMs. Specifically, (1) on the data side, we design an automated preference data construction pipeline to create [MMPR](https://huggingface.co/datasets/OpenGVLab/MMPR), a high-quality, large-scale multimodal reasoning preference dataset. and (2) on the model side, we explore integrating PO with MLLMs, developing a simple yet effective method, termed Mixed Preference Optimization (MPO), which boosts multimodal CoT performance.

Our approach demonstrates improved performance across multiple benchmarks, particularly in multimodal reasoning tasks. Notably, our model, [InternVL2-8B-MPO](https://huggingface.co/OpenGVLab/InternVL2-8B), achieves an accuracy of 67.0 on MathVista, outperforming InternVL2-8B by 8.7 points and achieving performance comparable to the 10$\times$ larger InternVL2-76B. We hope this study could inspire further advancements in MLLMs.


## Model Details

InternVL2-8B-MPO is initialized from [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B) and finetuned using [MMPR](https://huggingface.co/datasets/OpenGVLab/MMPR), a large-scale multimodal reasoning preference dataset.
This model exhibits enhanced multimodal reasoning abilities and fewer hallucinations compared to InternVL2-8B.

## Performance

| Model Name              | M3CoT | MathVista | MathVision MINI | MMVet (GPT4-Turbo) | LLaVA-Bench | POPE  | CRPE  | MMHalBench |
| ----------------------- | :---: | :-------: | :-------------: | :----------------: | :---------: | :---: | :---: | :--------: |
| Gemini-1.5-Pro          |   -   |   63.9    |      19.2       |         -          |      -      |   -   |   -   |     -      |
| GPT-4o                  | 64.3  |   63.8    |      30.4       |        69.1        |    97.6     | 86.9  | 76.6  |    4.0     |
| GPT-4o-Mini             | 61.9  |   52.4    |      27.3       |        66.9        |    95.4     | 85.1  | 73.1  |    3.6     |
| LLaVA-1.5-13B           | 39.5  |   27.6    |      11.1       |        36.3        |    70.7     | 85.9  | 55.6  |    2.4     |
| Qwen2-VL-7B             | 57.8  |   58.2    |      21.1       |        60.6        |    67.7     | 88.1  | 74.4  |    3.4     |
| MiniCPM-V-2-6-8B        | 56.0  |   60.6    |      23.4       |        57.4        |    83.4     | 87.3  | 75.2  |    3.6     |
| LLaVA-OneVision-7B      | 52.3  |   63.2    |      18.4       |        51.4        |    79.9     | 88.4  | 73.7  |    3.1     |
| InternVL2-26B           | 58.2  |   59.4    |      23.4       |        62.1        |    92.3     | 88.0  | 75.6  |    3.7     |
| InternVL2-40B           | 63.6  |   63.7    |      21.4       |        65.5        |    100.5    | 88.4  | 77.3  |    3.9     |
| InternVL2-76B           | 65.4  |   67.5    |      23.7       |        65.7        |    99.3     | 89.0  | 77.8  |    3.8     |
| InternVL2-Pro           | 65.6  |   66.3    |      18.8       |        69.4        |    99.5     | 88.2  | 77.6  |    3.7     |
| InternVL2-8B            | 59.3  |   58.3    |      20.4       |        54.2        |    73.2     | 86.9  | 75.0  |    3.3     |
| InternVL2-8B-MPO (ours) | 79.2  |   67.0    |      25.7       |        56.2        |    76.7     | 88.1  | 75.4  |    3.5     |

### Invitation to Evaluate InternVL
We welcome MLLM benchmark developers to assess our InternVL1.5 and InternVL2 series models. If you need to add your evaluation results here, please contact me at [wangweiyun@pjlab.org.cn](mailto:wangweiyun@pjlab.org.cn).
## Quick Start
We provide an example code to run InternVL2-8B using `transformers`.
We also welcome you to experience the InternVL2 series models in our [online demo](https://internvl.opengvlab.com/).
> Please use transformers==4.37.2 to ensure the model works normally.
### Model Loading
#### 16-bit (bf16 / fp16)
```python
import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2-8B-MPO"
model = AutoModel.from_pretrained(
    path,
    dtype=paddle.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```
#### BNB 8-bit Quantization

```python
import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2-8B-MPO"
model = AutoModel.from_pretrained(
    path,
    dtype=paddle.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

#### BNB 4-bit Quantization

```python
import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2-8B-MPO"
model = AutoModel.from_pretrained(
    path,
    dtype=paddle.bfloat16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

#### Multiple GPUs

The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors.

```python
import math
import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModel
def split_model(model_name):
    device_map = {}
    world_size = paddle.device.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map
path = "OpenGVLab/InternVL2-8B-MPO"
device_map = split_model('InternVL2-8B')
model = AutoModel.from_pretrained(
    path,
    dtype=paddle.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```

### Inference with Transformers

```python
import numpy as np
import paddle
import paddlevision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from paddlenlp.transformers import AutoModel, AutoTokenizer
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = paddle.stack(pixel_values)
    return pixel_values
# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OpenGVLab/InternVL2-8B-MPO'
model = AutoModel.from_pretrained(
    path,
    dtype=paddle.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(paddle.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)
# pure-text conversation (纯文本对话)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')
question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
# single-image single-round conversation (单图单轮对话)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')
# single-image multi-round conversation (单图多轮对话)
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')
question = 'Please write a poem according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
# multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(paddle.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(paddle.bfloat16).cuda()
pixel_values = paddle.concat((pixel_values1, pixel_values2), dim=0)
question = '<image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')
question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
# multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(paddle.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(paddle.bfloat16).cuda()
pixel_values = paddle.concat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')
question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
# batch inference, single image per sample (单图批处理)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(paddle.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(paddle.bfloat16).cuda()
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
pixel_values = paddle.concat((pixel_values1, pixel_values2), dim=0)
questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')
# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices
def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = paddle.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = paddle.concat(pixel_values_list)
    return pixel_values, num_patches_list
video_path = './examples/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
pixel_values = pixel_values.to(paddle.bfloat16).cuda()
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
question = video_prefix + 'What is the red panda doing?'
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')
question = 'Describe this video in detail. Don\'t repeat.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
```

#### Streaming output

Besides this method, you can also use the following code to get streamed output.

```python
from paddlenlp.transformers import TextIteratorStreamer
from threading import Thread
# Initialize the streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
# Define the generation configuration
generation_config = dict(max_new_tokens=1024, do_sample=False, streamer=streamer)
# Start the model chat in a separate thread
thread = Thread(target=model.chat, kwargs=dict(
    tokenizer=tokenizer, pixel_values=pixel_values, question=question,
    history=None, return_history=False, generation_config=generation_config,
))
thread.start()
# Initialize an empty string to store the generated text
generated_text = ''
# Loop through the streamer to get the new text as it is generated
for new_text in streamer:
    if new_text == model.conv_template.sep:
        break
    generated_text += new_text
    print(new_text, end='', flush=True)  # Print each new chunk of generated text on the same line
```

## Finetune

Many repositories now support fine-tuning of the InternVL series models, including [InternVL](https://github.com/OpenGVLab/InternVL), [SWIFT](https://github.com/modelscope/ms-swift), [XTurner](https://github.com/InternLM/xtuner), and others. Please refer to their documentation for more details on fine-tuning.

For preference optimization, you can refer to[ this document](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html).

## Deployment

### LMDeploy

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the MMRazor and MMDeploy teams.

```sh
pip install lmdeploy==0.5.3
```

LMDeploy abstracts the complex inference process of multi-modal Vision-Language Models (VLM) into an easy-to-use pipeline, similar to the Large Language Model (LLM) inference pipeline.

#### A 'Hello, world' example

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
model = 'OpenGVLab/InternVL2-8B-MPO'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

If `ImportError` occurs while executing this case, please install the required dependency packages as prompted.

#### Multi-images inference

When dealing with multiple images, you can put them all in one list. Keep in mind that multiple images will lead to a higher number of input tokens, and as a result, the size of the context window typically needs to be increased.

> Warning: Due to the scarcity of multi-image conversation data, the performance on multi-image tasks may be unstable, and it may require multiple attempts to achieve satisfactory results.

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
model = 'OpenGVLab/InternVL2-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]
images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

#### Batch prompts inference

Conducting inference with batch prompts is quite straightforward; just place them within a list structure:

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
model = 'OpenGVLab/InternVL2-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

#### Multi-turn conversation

There are two ways to do the multi-turn conversations with the pipeline. One is to construct messages according to the format of OpenAI and use above introduced method, the other is to use the `pipeline.chat` interface.

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
model = 'OpenGVLab/InternVL2-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

#### Service

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-8B-MPO --backend turbomind --server-port 23333
```

To use the OpenAI-style interface, you need to install OpenAI:

```shell
pip install openai
```

Then, use the code below to make the API call:

```python
from openai import OpenAI
client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```

## License

This project is released under the MIT license, while InternLM2 is licensed under the Apache-2.0 license.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

## 简介

现有的开源多模态大语言模型（Multimodal Large Language Models, MLLMs）通常采用包括预训练和监督微调的训练过程。然而，这些模型往往面临分布偏移问题，限制了其多模态推理能力，尤其是在Chain-of-Thought（CoT）推理性能方面。

为了解决这一问题，我们引入了额外的偏好优化过程，以增强MLLMs的多模态链式推理能力。具体而言，（1）在数据方面，我们设计了一个自动化的偏好数据构建管线，并构建了MMPR，一个高质量、大规模的多模态推理偏好数据集；（2）在模型方面，我们探索了如何将偏好优化过程整合到现有的MLLM中，开发了一种简单而有效的方法，称为混合偏好优化（Mixed Preference Optimization, MPO），显著提升了多模态CoT表现。

我们的方法在多个基准上展现了卓越的效果，尤其是在多模态推理任务中，相比InternVL2-8B性能显著提升。
值得注意的是，我们的模型InternVL2-8B-MPO在MathVista上取得了67.0%的准确率，超越InternVL2-8B 8.7个点，且表现接近于大10倍的InternVL2-76B。我们希望本研究能够促进MLLMs的进一步发展。

## 模型细节

InternVL2-8B-MPO基于[InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B)初始化，并使用[MMPR](https://huggingface.co/datasets/OpenGVLab/MMPR)这一大规模多模态推理偏好数据集进行微调。与InternVL2-8B相比，该模型表现出更强的多模态推理能力，且幻觉现象更少。

## 性能测试

| Model Name              | M3CoT | MathVista | MathVision MINI | MMVet (GPT4-Turbo) | LLaVA-Bench | POPE  | CRPE  | MMHalBench |
| ----------------------- | :---: | :-------: | :-------------: | :----------------: | :---------: | :---: | :---: | :--------: |
| Gemini-1.5-Pro          |   -   |   63.9    |      19.2       |         -          |      -      |   -   |   -   |     -      |
| GPT-4o                  | 64.3  |   63.8    |      30.4       |        69.1        |    97.6     | 86.9  | 76.6  |    4.0     |
| GPT-4o-Mini             | 61.9  |   52.4    |      27.3       |        66.9        |    95.4     | 85.1  | 73.1  |    3.6     |
| LLaVA-1.5-13B           | 39.5  |   27.6    |      11.1       |        36.3        |    70.7     | 85.9  | 55.6  |    2.4     |
| Qwen2-VL-7B             | 57.8  |   58.2    |      21.1       |        60.6        |    67.7     | 88.1  | 74.4  |    3.4     |
| MiniCPM-V-2-6-8B        | 56.0  |   60.6    |      23.4       |        57.4        |    83.4     | 87.3  | 75.2  |    3.6     |
| LLaVA-OneVision-7B      | 52.3  |   63.2    |      18.4       |        51.4        |    79.9     | 88.4  | 73.7  |    3.1     |
| InternVL2-26B           | 58.2  |   59.4    |      23.4       |        62.1        |    92.3     | 88.0  | 75.6  |    3.7     |
| InternVL2-40B           | 63.6  |   63.7    |      21.4       |        65.5        |    100.5    | 88.4  | 77.3  |    3.9     |
| InternVL2-76B           | 65.4  |   67.5    |      23.7       |        65.7        |    99.3     | 89.0  | 77.8  |    3.8     |
| InternVL2-Pro           | 65.6  |   66.3    |      18.8       |        69.4        |    99.5     | 88.2  | 77.6  |    3.7     |
| InternVL2-8B            | 59.3  |   58.3    |      20.4       |        54.2        |    73.2     | 86.9  | 75.0  |    3.3     |
| InternVL2-8B-MPO (ours) | 79.2  |   67.0    |      25.7       |        56.2        |    76.7     | 88.1  | 75.4  |    3.5     |

### 邀请评测 InternVL
我们欢迎各位 MLLM benchmark 的开发者对我们的 InternVL2-8B-MPO 模型进行评测。如果需要在此处添加评测结果，请与我联系（[wangweiyun@pjlab.org.cn](mailto:wangweiyun@pjlab.org.cn)）。
## 快速启动
我们提供了一个示例代码，用于使用 `transformers` 运行 InternVL2-8B。
我们也欢迎你在我们的[在线demo](https://internvl.opengvlab.com/)中体验InternVL2的系列模型。
> 请使用 transformers==4.37.2 以确保模型正常运行。
示例代码请[点击这里](#quick-start)。
## 微调
许多仓库现在都支持 InternVL 系列模型的微调，包括 [InternVL](https://github.com/OpenGVLab/InternVL)、[SWIFT](https://github.com/modelscope/ms-swift)、[XTurner](https://github.com/InternLM/xtuner) 等。请参阅它们的文档以获取更多微调细节。

如果希望基于偏好对齐进行模型训练，可以参考这份[文档](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html)。

## 部署
### LMDeploy
LMDeploy 是由 MMRazor 和 MMDeploy 团队开发的用于压缩、部署和服务大语言模型（LLM）的工具包。
```sh
pip install lmdeploy==0.5.3
```
LMDeploy 将多模态视觉-语言模型（VLM）的复杂推理过程抽象为一个易于使用的管道，类似于大语言模型（LLM）的推理管道。
#### 一个“你好，世界”示例
```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-8B-MPO'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```
如果在执行此示例时出现 `ImportError`，请按照提示安装所需的依赖包。
#### 多图像推理
在处理多张图像时，可以将它们全部放入一个列表中。请注意，多张图像会导致输入 token 数量增加，因此通常需要增加上下文窗口的大小。
```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
model = 'OpenGVLab/InternVL2-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]
images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```
#### 批量Prompt推理
使用批量Prompt进行推理非常简单；只需将它们放在一个列表结构中：
```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
model = 'OpenGVLab/InternVL2-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```
#### 多轮对话

使用管道进行多轮对话有两种方法。一种是根据 OpenAI 的格式构建消息并使用上述方法，另一种是使用 `pipeline.chat` 接口。

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
model = 'OpenGVLab/InternVL2-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

#### API部署

LMDeploy 的 `api_server` 使模型能够通过一个命令轻松打包成服务。提供的 RESTful API 与 OpenAI 的接口兼容。以下是服务启动的示例：

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-8B-MPO --backend turbomind --server-port 23333
```

为了使用OpenAI风格的API接口，您需要安装OpenAI:

```shell
pip install openai
```

然后，使用下面的代码进行API调用:

```python
from openai import OpenAI
client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```

## 开源许可证

该项目采用 MIT 许可证发布，而 InternLM2 则采用 Apache-2.0 许可证。

## 引用

如果您发现此项目对您的研究有用，可以考虑引用我们的论文：

```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/README.md) (34.4 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/added_tokens.json) (179.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/config.json) (5.3 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/generation_config.json) (34.0 B)

- [model-00001-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/model-00001-of-00004.safetensors) (4.6 GB)

- [model-00002-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/model-00002-of-00004.safetensors) (4.6 GB)

- [model-00003-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/model-00003-of-00004.safetensors) (4.6 GB)

- [model-00004-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/model-00004-of-00004.safetensors) (1.3 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/model.safetensors.index.json) (50.0 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/special_tokens_map.json) (844.0 B)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/tokenizer.model) (1.4 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/tokenizer_config.json) (3.9 KB)

- [train_results.json](https://paddlenlp.bj.bcebos.com/models/community/OpenGVLab/InternVL2-8B-MPO/train_results.json) (198.0 B)


[Back to Main](../../)