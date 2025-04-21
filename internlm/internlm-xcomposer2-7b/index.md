
# internlm-xcomposer2-7b
---


## README([From Huggingface](https://huggingface.co/internlm/internlm-xcomposer2-7b))




<p align="center">
    <img src="logo_en.png" width="400"/>
<p>

<p align="center">
    <b><font size="6">InternLM-XComposer2</font></b> 
<p>

<div align="center">

[💻Github Repo](https://github.com/InternLM/InternLM-XComposer)

[Paper](https://arxiv.org/abs/2401.16420)

</div>

**InternLM-XComposer2** is a vision-language large model (VLLM) based on [InternLM2](https://github.com/InternLM/InternLM) for advanced text-image comprehension and composition. 

We release InternLM-XComposer2 series in two versions:

- InternLM-XComposer2-VL: The pretrained VLLM model with InternLM2 as the initialization of the LLM, achieving strong performance on various multimodal benchmarks.
- InternLM-XComposer2: The finetuned VLLM for *Free-from Interleaved Text-Image Composition*.

### Import from Transformers
To load the InternLM-XComposer2-7B model using Transformers, use the following code:
```python
import paddle
from PIL import Image
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_path = "internlm/internlm-xcomposer2-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=paddle.float32, trust_remote_code=True).cuda()
# Set `dtype=paddle.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
# model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=paddle.float16, trust_remote_code=True).cuda()
model = model.eval()
img_path_list = [
    './panda.jpg',
    './bamboo.jpeg',
]
images = []
for img_path in img_path_list:
    image = Image.open(img_path).convert("RGB")
    image = model.vis_processor(image)
    images.append(image)
image = paddle.stack(images)
query = '<ImageHere> <ImageHere>please write an article based on the images. Title: my favorite animal.'
with torch.cuda.amp.autocast():
    response, history = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
print(response)
""""
# My favorite animal is the panda. Pandas are one of the most popular animals in the world, and for good reason. They are adorable, cuddly creatures that have captured the hearts of people all over the globe.
Pandas are native to China and can be found in the wild in a few specific regions. However, they are also very popular in captivity, with many zoos around the world housing pandas as part of their exhibits. I have been fortunate enough to see pandas up close at several different zoos, and each time it was an amazing experience.
One thing that always strikes me about pandas is how much they love to eat bamboo. In fact, pandas spend almost all of their waking hours eating bamboo. This may not seem like a lot of fun, but pandas actually enjoy munching on this tough plant. It's fascinating to watch them chew through the tough stalks and leaves, and then lick their lips in satisfaction.
Another thing that I find interesting about pandas is their black and white fur. The combination of these two colors creates a striking contrast that makes pandas instantly recognizable. In addition, the black patches on their face give them a unique expression that seems to convey both playfulness and seriousness.
Despite their popularity, pandas do face some challenges. Their habitat is being destroyed by human activities such as logging and agriculture, which has led to a decline in their population. Additionally, pandas are considered endangered due to factors such as low reproductive rates and limited genetic diversity.
However, there are efforts underway to protect pandas and their habitats. Many organizations work to raise awareness about the importance of preserving these beautiful creatures, and governments in countries where pandas live are taking steps to conserve their natural environment.
In conclusion, pandas are truly remarkable animals that deserve our admiration and protection. With their distinctive appearance, playful personalities, and love of bamboo, it's no wonder that pandas have become so beloved around the world. Let's do what we can to ensure that future generations can continue to appreciate these wonderful creatures.
"""
```

### 通过 Transformers 加载
通过以下的代码加载 InternLM-XComposer2-7B 模型

```python
import paddle
from PIL import Image
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_path = "internlm/internlm-xcomposer2-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=paddle.float32, trust_remote_code=True).cuda()
# `dtype=paddle.float16` 可以令模型以 float16 精度加载，否则 transformers 会将模型加载为 float32，导致显存不足
# model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=paddle.float16, trust_remote_code=True).cuda()
model = model.eval() 
img_path_list = [
    './panda.jpg',
    './bamboo.jpeg',
]
images = []
for img_path in img_path_list:
    image = Image.open(img_path).convert("RGB")
    image = model.vis_processor(image)
    images.append(image)
image = paddle.stack(images)
query = '<ImageHere> <ImageHere>请根据图片写一篇作文：我最喜欢的小动物。要求：选准角度，确定立意，明确文体，自拟标题。'
with torch.cuda.amp.autocast():
    response, history = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
print(response)
"""
# 我最喜欢的小动物
我喜欢的动物有很多，有活泼可爱的小狗、美丽高贵的孔雀、凶猛的狮子……但我最喜欢的是憨态可掬的大熊猫。
大熊猫是国宝，它有着黑白相间的毛色，圆滚滚的身体，胖乎乎的手脚，大大的眼睛和短短的尾巴。它的耳朵小小的，像两片树叶；嘴巴又宽又扁，就像一个“月牙”；四肢短小粗壮，走起路来摇摇晃晃，非常可爱。
大熊猫喜欢吃竹子，每天要吃30多斤呢！它们吃竹子的样子很特别，先把竹子咬断，然后抱着竹子啃起来，有时还会把竹子扔到空中再接住继续啃，好像在表演杂技一样。吃饱了以后，它们就懒洋洋地躺在地上睡大觉，真是个名副其实的“大懒猫”啊！
大熊猫不仅爱吃竹子，还爱睡觉。一天中，除了吃饭的时间，其他时间都在睡觉。有时候，它们会爬上树，坐在树枝上呼呼大睡；有时候，它们会找一个阴凉的地方，躺下来美美地睡上一觉。
大熊猫还是一种濒危动物，因为它们的栖息地被破坏，食物减少，数量越来越少。为了保护大熊猫，人们建立了大熊猫保护区，禁止砍伐树木，让大熊猫有一个安全的家。
我喜欢大熊猫，因为它既可爱又珍贵，我希望它能一直生活在我们的地球上，陪伴着我们成长。
"""
```

### Open Source License
The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow free commercial usage. To apply for a commercial license, please fill in the application form (English)/申请表（中文）. For other questions or collaborations, please contact internlm@pjlab.org.cn.




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/internlm/internlm-xcomposer2-7b/README.md) (7.1 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/internlm/internlm-xcomposer2-7b/config.json) (1.1 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/internlm/internlm-xcomposer2-7b/generation_config.json) (175.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/internlm/internlm-xcomposer2-7b/model_state.pdparams) (32.3 GB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/internlm/internlm-xcomposer2-7b/sentencepiece.bpe.model) (1.4 MB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/internlm/internlm-xcomposer2-7b/special_tokens_map.json) (95.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/internlm/internlm-xcomposer2-7b/tokenizer_config.json) (401.0 B)


[Back to Main](../../)