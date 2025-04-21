
# Randeng-BART-139M-SUMMARY
---


## README([From Huggingface](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M-SUMMARY))

---
language: 
  - zh
license: apache-2.0

inference: true

widget:
- text: 'summary: 在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！'
---

# Randeng-BART-139M-SUMMARY

- Main Page:[Fengshenbang](https://fengshenbang-lm.com/)
- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)

## 简介 Brief Introduction

善于处理摘要任务，在一个中文摘要数据集上微调后的，中文版的BART-base。

Good at solving text summarization tasks, after fine-tuning on a Chinese text summarization dataset, Chinese BART-base.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | BART |      139M      |     中文-文本摘要任务 Chinese-Summary    |

## 模型信息 Model Information

基于[Randeng-BART-139M](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M)，我们在收集的1个中文领域的文本摘要数据集（LCSTS）上微调了它，得到了summary版本。

Based on 基于[Randeng-BART-139M](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M), we fine-tuned a text summarization version (summary) on a Chinese text summarization datasets (LCSTS).

## 使用 Usage

```python
from paddlenlp.transformers import BartForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline
import paddle

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Randeng-BART-139M-SUMMARY')
model=BartForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-BART-139M-SUMMARY')
text = 'summary:在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！'
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
print(text2text_generator(text, max_length=50, do_sample=False))
```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):

```text
@article{fengshenbang,
  author    = {Jiaxing Zhang and Ruyi Gan and Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

也可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Randeng-BART-139M-SUMMARY/README.md) (4.4 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Randeng-BART-139M-SUMMARY/added_tokens.json) (33.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Randeng-BART-139M-SUMMARY/config.json) (1.2 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Randeng-BART-139M-SUMMARY/model_state.pdparams) (486.9 MB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Randeng-BART-139M-SUMMARY/special_tokens_map.json) (157.0 B)

- [spiece.model](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Randeng-BART-139M-SUMMARY/spiece.model) (838.4 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Randeng-BART-139M-SUMMARY/tokenizer_config.json) (419.0 B)


[Back to Main](../../)