
# TinyLlama-1.1B-intermediate-step-240k-503b
---


## README([From Huggingface](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b))


<div align="center">

# TinyLlama-1.1B
</div>

https://github.com/jzhang38/TinyLlama

The TinyLlama project aims to **pretrain** a **1.1B Llama model on 3 trillion tokens**. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01. 

<div align="center">
  <img src="./TinyLlama_logo.png" width="300"/>
</div>

We adopted exactly the same architecture and tokenizer as Llama 2. This means TinyLlama can be plugged and played in many open-source projects built upon Llama. Besides, TinyLlama is compact with only 1.1B parameters. This compactness allows it to cater to a multitude of applications demanding a restricted computation and memory footprint.

#### This Model
This is an intermediate checkpoint with 240K steps and 503B tokens. **We suggest you not use this directly for inference.** The [chat model](https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.1) is always preferred **


#### How to use
You will need the transformers>=4.31
Do check the [TinyLlama](https://github.com/jzhang38/TinyLlama) github page for more information.
```
from paddlenlp.transformers import AutoTokenizer
import transformers 
import torch
model = "PY007/TinyLlama-1.1B-intermediate-step-240k-503b"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    dtype=paddle.float16,
)

sequences = pipeline(
    'The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01.',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    repetition_penalty=1.5,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/README.md) (2.0 KB)

- [TinyLlama_logo.png](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/TinyLlama_logo.png) (1.8 MB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/config.json) (601.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/generation_config.json) (129.0 B)

- [model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/model.safetensors) (4.1 GB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/special_tokens_map.json) (414.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/tokenizer.json) (1.8 MB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/tokenizer.model) (488.0 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b/tokenizer_config.json) (776.0 B)


[Back to Main](../../)