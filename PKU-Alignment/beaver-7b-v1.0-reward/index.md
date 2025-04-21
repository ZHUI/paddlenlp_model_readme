
# beaver-7b-v1.0-reward
---


## README([From Huggingface](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-reward))

---
datasets:
  - PKU-Alignment/PKU-SafeRLHF
language:
  - en
tags:
  - reinforcement-learning-from-human-feedback
  - reinforcement-learning
  - beaver
  - safety
  - llama
  - ai-safety
  - deepspeed
  - rlhf
  - alpaca
library_name: safe-rlhf
---

# 🦫 Beaver's Reward Model

## Model Details

The Beaver reward model is a preference model trained using the [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) dataset.
It can play a role in the safe RLHF algorithm, helping the Beaver model become more helpful.

- **Developed by:** the [PKU-Alignment](https://github.com/PKU-Alignment) Team.
- **Model Type:** An auto-regressive language model based on the transformer architecture.
- **License:** Non-commercial license.
- **Fine-tuned from model:** [LLaMA](https://arxiv.org/abs/2302.13971), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Model Sources

- **Repository:** <https://github.com/PKU-Alignment/safe-rlhf>
- **Beaver:** <https://huggingface.co/PKU-Alignment/beaver-7b-v1.0>
- **Dataset:** <https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF>
- **Reward Model:** <https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-reward>
- **Cost Model:** <https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-cost>
- **Dataset Paper:** <https://arxiv.org/abs/2307.04657>
- **Paper:** <https://arxiv.org/abs/2310.12773>

## How to Use the Reward Model

```python
import torch
from paddlenlp.transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore

model = AutoModelForScore.from_pretrained('PKU-Alignment/beaver-7b-v1.0-reward', dtype=paddle.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('PKU-Alignment/beaver-7b-v1.0-reward')

input = 'BEGINNING OF CONVERSATION: USER: hello ASSISTANT:Hello! How can I help you today?'

input_ids = tokenizer(input, return_tensors='pt')
output = model(**input_ids)
print(output)

# ScoreModelOutput(
#     scores=tensor([[[-19.7500],
#          [-19.3750],
#          [-20.1250],
#          [-18.0000],
#          [-20.0000],
#          [-23.8750],
#          [-23.5000],
#          [-22.0000],
#          [-21.0000],
#          [-20.1250],
#          [-23.7500],
#          [-21.6250],
#          [-21.7500],
#          [-12.9375],
#          [ -6.4375],
#          [ -8.1250],
#          [ -7.3438],
#          [ -9.1875],
#          [-13.6250],
#          [-10.5625],
#          [ -9.9375],
#          [ -6.4375],
#          [ -6.0938],
#          [ -5.8438],
#          [ -6.6562],
#          [ -5.9688],
#          [ -9.1875],
#          [-11.4375]]], grad_fn=<ToCopyBackward0>),
#     end_scores=tensor([[-11.4375]], grad_fn=<ToCopyBackward0>),
#     last_hidden_state=tensor([[[ 0.7461, -0.6055, -0.4980,  ...,  0.1670,  0.7812, -0.3242],
#          [ 0.7383, -0.5391, -0.1836,  ..., -0.1396,  0.5273, -0.2256],
#          [ 0.6836, -0.7031, -0.3730,  ...,  0.2100,  0.5000, -0.6328],
#          ...,
#          [-1.7969,  1.0234,  1.0234,  ..., -0.8047,  0.2500, -0.8398],
#          [ 2.0469, -1.3203,  0.8984,  ..., -0.7734, -1.4141, -1.6797],
#          [ 4.3438, -0.6953,  0.9648,  ..., -0.1787,  0.6680, -3.0000]]],
#        dtype=paddle.bfloat16, grad_fn=<ToCopyBackward0>),
#     end_last_hidden_state=tensor([[ 4.3438, -0.6953,  0.9648,  ..., -0.1787,  0.6680, -3.0000]],
#        dtype=paddle.bfloat16, grad_fn=<ToCopyBackward0>),
#     end_index=tensor([27])
# )
```




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/README.md) (3.3 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/added_tokens.json) (21.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/config.json) (1.1 KB)

- [model_state-00001-of-00002.pdparams](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/model_state-00001-of-00002.pdparams) (9.3 GB)

- [model_state-00002-of-00002.pdparams](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/model_state-00002-of-00002.pdparams) (3.0 GB)

- [model_state.pdparams.index.json](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/model_state.pdparams.index.json) (24.4 KB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/sentencepiece.bpe.model) (488.0 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/special_tokens_map.json) (549.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/PKU-Alignment/beaver-7b-v1.0-reward/tokenizer_config.json) (1.1 KB)


[Back to Main](../../)