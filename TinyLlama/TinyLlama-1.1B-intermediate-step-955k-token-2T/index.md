
# TinyLlama-1.1B-intermediate-step-955k-token-2T
---


## README([From Huggingface](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T))


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
This is an intermediate checkpoint with 995K steps and 2003B tokens.

#### Releases Schedule
We will be rolling out intermediate checkpoints following the below schedule. We also include some baseline models for comparison.

| Date       | HF Checkpoint                                   | Tokens | Step | HellaSwag Acc_norm |
|------------|-------------------------------------------------|--------|------|---------------------|
| Baseline   | [StableLM-Alpha-3B](https://huggingface.co/stabilityai/stablelm-base-alpha-3b)| 800B   | --   |  38.31            |
| Baseline   | [Pythia-1B-intermediate-step-50k-105b](https://huggingface.co/EleutherAI/pythia-1b/tree/step50000)             | 105B   | 50k   |  42.04            |
| Baseline   | [Pythia-1B](https://huggingface.co/EleutherAI/pythia-1b)             | 300B   | 143k   |  47.16            |
| 2023-09-04 | [TinyLlama-1.1B-intermediate-step-50k-105b](https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b) | 105B   | 50k   |  43.50               |
| 2023-09-16 | --                                             | 500B   | --   |  --               |
| 2023-10-01 | --                                             | 1T     | --   |  --               |
| 2023-10-16 | --                                             | 1.5T   | --   |  --               |
| 2023-10-31 | --                                             | 2T     | --   |  --               |
| 2023-11-15 | --                                             | 2.5T   | --   |  --               |
| 2023-12-01 | --                                             | 3T     | --   |  --               |

#### How to use
You will need the transformers>=4.31
Do check the [TinyLlama](https://github.com/jzhang38/TinyLlama) github page for more information.
```
from transformers import AutoTokenizer
import transformers 
import torch
model = "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
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

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/README.md) (3.4 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/config.json) (554.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/generation_config.json) (129.0 B)

- [model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/model.safetensors) (4.1 GB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/special_tokens_map.json) (414.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/tokenizer.json) (1.8 MB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/tokenizer.model) (488.0 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/tokenizer_config.json) (776.0 B)


[Back to Main](../../)