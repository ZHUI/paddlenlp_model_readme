
# mamba-1.4b-hf
---


## README([From Huggingface](https://huggingface.co/state-spaces/mamba-1.4b-hf))

---
library_name: transformers
tags: []
---

# Mamba

<!-- Provide a quick summary of what the model is/does. -->
This repository contains the `transfromers` compatible `mamba-2.8b`. The checkpoints are untouched, but the full `config.json` and tokenizer are pushed to this repo. 

# Usage

You need to install `transformers` from `main` until `transformers=4.39.0` is released. 
```bash
pip install git+https://github.com/huggingface/transformers@main
```

We also recommend you to install both `causal_conv_1d` and `mamba-ssm` using: 

```bash
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

If any of these two is not installed, the "eager" implementation will be used. Otherwise the more optimised `cuda` kernels will be used.

## Generation
You can use the classic `generate` API:
```python
>>> from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
>>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")
>>> input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]

>>> out = model.generate(input_ids, max_new_tokens=10)
>>> print(tokenizer.batch_decode(out))
["Hey how are you doing?\n\nI'm doing great.\n\nI"]
```

## PEFT finetuning example
In order to finetune using the `peft` library, we recommend keeping the model in float32!

```python
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")
dataset = load_dataset("Abirate/english_quotes", split="train")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)
trainer.train()
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/README.md) (2.3 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/added_tokens.json) (552.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/config.json) (872.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/generation_config.json) (96.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/merges.txt) (445.9 KB)

- [model-00001-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/model-00001-of-00002.safetensors) (4.6 GB)

- [model-00002-of-00002.safetensors](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/model-00002-of-00002.safetensors) (504.0 MB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/model.safetensors.index.json) (37.3 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/special_tokens_map.json) (3.6 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/tokenizer_config.json) (3.8 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/state-spaces/mamba-1.4b-hf/vocab.json) (976.4 KB)


[Back to Main](../../)