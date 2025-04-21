
# glm-lite-pro
---


## README([From Huggingface](https://huggingface.co/THUDM/glm-lite-pro))

---
support_training: 0
 
license: Apache License 2.0

---



#### Clone with HTTP
在个人中心->模型->我的模型，查询访问令牌。可以通过令牌进行git仓库的使用。
```bash
 git clone http://git.aistudio.baidu.com/16005791/glm-lite-pro.git
```


## glm-lite-pro
paddle框架适配版本，支持预训练、SFT、Lora
详见finetune.md

### 环境准备
- PaddlePaddle 3.0-beta
- PaddleNLP 3.0.0b3

### 快速开始
代码块
```
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
path = ""

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype="float16")

input_features = tokenizer("China is planning to reduce their greenhouse gas emissions by implementing new policies in the transportation and industrial sectors. What proposals do you think China could implement to achieve their emission-reduction goals?", return_tensors="pd")
outputs = model.generate(**input_features, max_length=1024)
print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
```

## 评测
首先从 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e84444333b6d434ea7b0) 下载处理好的 C-Eval 数据集，解压到 `evaluation` 目录下。然后运行

```shell
import os
import glob
import re
import json


import paddle
import paddlenlp
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
paddle.set_device('gpu:1')

from tqdm import tqdm
model_path=""

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype="float16")

choices = ["A", "B", "C", "D"]
#choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]
choice_tokens = [tokenizer(choice, add_special_tokens=False, return_tensors="pd")["input_ids"][0].tolist()[0] for choice in choices]


def build_prompt(text):
    return "[Round {}]\n\n问：{}\n\n答：".format(1, text)


extraction_prompt = '综上所述，ABCD中正确的选项是：'

accuracy_dict, count_dict = {}, {}
with paddle.no_grad():
    for entry in glob.glob("./CEval/val/**/*.jsonl", recursive=True):
        dataset = []
        with open(entry, encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line))
        correct = 0
        dataloader = paddle.io.DataLoader(dataset, batch_size=8)
        for batch in tqdm(dataloader):
            texts = batch["inputs_pretokenized"]
            queries = [build_prompt(query) for query in texts]
            
            inputs = tokenizer(queries, return_tensors="pd", padding=True,truncation=True, max_length=2048)
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            intermediate_outputs = []
            response=tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
            intermediate_outputs=response

            answer_texts = [text + intermediate + "\n" + extraction_prompt for text, intermediate in
                            zip(texts, intermediate_outputs)]
            input_tokens = [build_prompt(answer_text) for answer_text in answer_texts]

            inputs = tokenizer(input_tokens, padding=True, return_tensors="pd", truncation=True, max_length=2048)
            outputs = model(**inputs, return_last_logit=True)


            logits = outputs[0][:,-1]
            logits = logits[:, choice_tokens]
            preds = logits.argmax(axis=-1)
            correct += (preds.cpu() == batch["label"]).sum().item()
        accuracy = correct / len(dataset)
        print(entry, accuracy)
        accuracy_dict[entry] = accuracy
        count_dict[entry] = len(dataset)

acc_total, count_total = 0.0, 0
for key in accuracy_dict:
    acc_total += accuracy_dict[key] * count_dict[key]
    count_total += count_dict[key]
print(acc_total / count_total)
```

这个脚本会在C-Eval的验证集上进行预测并输出准确率。如果想要得到测试集上的结果可以将代码中的 `model_path = ""` 改为 `model_path ="./CEval/test/**/*.jsonl"`，并按照 C-Eval 规定的格式保存结果并在 [官网](https://cevalbenchmark.com/) 上提交。

汇报的结果使用的是内部的并行测试框架，结果可能会有轻微波动。



## 关于模型
本模型的推理及微调与训练适配是model_state.pdparams权重格式下做的，由于上传文件大小限制，上传的模型文件转换为例safetensors格式。如有问题欢迎咨询。



## Model Files

- [.gitattributes](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/.gitattributes) (2.6 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/README.md) (4.4 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/config.json) (1.1 KB)

- [finetune.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/finetune.md) (13.2 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/generation_config.json) (124.0 B)

- [md5.info](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/md5.info) (540.0 B)

- [model-00001-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/model-00001-of-00004.safetensors) (3.6 GB)

- [model-00002-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/model-00002-of-00004.safetensors) (3.6 GB)

- [model-00003-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/model-00003-of-00004.safetensors) (3.6 GB)

- [model-00004-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/model-00004-of-00004.safetensors) (829.0 MB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/model.safetensors.index.json) (19.8 KB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/tokenizer.model) (994.5 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-lite-pro/tokenizer_config.json) (163.0 B)


[Back to Main](../../)