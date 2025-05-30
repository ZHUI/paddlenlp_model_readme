
# dialogue-bart-base-chinese
---


## README([From Huggingface](https://huggingface.co/HIT-TMG/dialogue-bart-base-chinese))

---
language: 
  - zh
thumbnail: "url to a thumbnail used in social sharing"
tags:
- bart-base-chinese
datasets:
- lccc
- kd_conv
---

# dialogue-bart-base-chinese
This is a seq2seq model fine-tuned on several Chinese dialogue datasets, from bart-base-chinese.


# Spaces
Now you can experience our model on HuggingFace Spaces [HIT-TMG/dialogue-bart-large-chinese](https://huggingface.co/spaces/HIT-TMG/dialogue-bart-large-chinese) .


# Datasets
We utilize 4 Chinese dialogue datasets from [LUGE](https://www.luge.ai/#/)

|                              |            |                       |
| ----                         | ----       | ----                  |
|                              | Count      | Domain                |
| Chinese Persona Chat (CPC)   | 23,000     | Open                  | 
| LCCC                         | 11,987,759 | Open                  |
| Emotional STC (ESTC)         | 899,207    | Open                  |
| KdConv                       | 3,000      | Movie, Music, Travel  |
|                              |            |                       |


# Data format
Input: `[CLS] 对话历史：<history> 知识：<knowledge> [SEP]`

Output: `[CLS] <response> [SEP]`


# Example
```python
from paddlenlp.transformers import BertTokenizer, BartForConditionalGeneration
# Note that tokenizer is an object of BertTokenizer, instead of BartTokenizer
tokenizer = BertTokenizer.from_pretrained("HIT-TMG/dialogue-bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("HIT-TMG/dialogue-bart-base-chinese")
# an example from CPC dev data
history = ["可以 认识 一下 吗 ？", "当然 可以 啦 ， 你好 。", "嘿嘿 你好 ， 请问 你 最近 在 忙 什么 呢 ？", "我 最近 养 了 一只 狗狗 ， 我 在 训练 它 呢 。"]
history_str = "对话历史：" + tokenizer.sep_token.join(history)
input_ids = tokenizer(history_str, return_tensors='pt').input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
 ```
 
 
 # Contact
 If you encounter any issue, feel free to contact us via the email: <u>yanshekwoo@foxmail.com</u>



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/HIT-TMG/dialogue-bart-base-chinese/README.md) (2.1 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/HIT-TMG/dialogue-bart-base-chinese/config.json) (1.7 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/HIT-TMG/dialogue-bart-base-chinese/model_state.pdparams) (443.6 MB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/HIT-TMG/dialogue-bart-base-chinese/special_tokens_map.json) (112.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/HIT-TMG/dialogue-bart-base-chinese/tokenizer_config.json) (449.0 B)

- [vocab.txt](https://paddlenlp.bj.bcebos.com/models/community/HIT-TMG/dialogue-bart-base-chinese/vocab.txt) (107.0 KB)


[Back to Main](../../)