
# Qwen2-1.5B-Instruct
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct))



# Qwen2-1.5B-Instruct

## Introduction

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This repo contains the instruction-tuned 1.5B Qwen2 model.

Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics, reasoning, etc.

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2/), [GitHub](https://github.com/QwenLM/Qwen2), and [Documentation](https://qwen.readthedocs.io/en/latest/).
<br>

## Model Details
Qwen2 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes.

## Training details
We pretrained the models with a large amount of data, and we post-trained the models with both supervised finetuning and direct preference optimization.


## Requirements
The code of Qwen2 has been in the latest Hugging face transformers and we advise you to install `transformers>=4.37.0`, or you might encounter the following error:
```
KeyError: 'qwen2'
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    
    
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pd")

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)[0]
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Evaluation

We briefly compare Qwen2-1.5B-Instruct with Qwen1.5-1.8B-Chat. The results are as follows:

| Datasets | Qwen1.5-0.5B-Chat | **Qwen2-0.5B-Instruct** | Qwen1.5-1.8B-Chat | **Qwen2-1.5B-Instruct** |
| :--- | :---: | :---: | :---: | :---: |
| MMLU | 35.0 | **37.9** | 43.7 | **52.4** |
| HumanEval | 9.1 | **17.1** | 25.0 | **37.8** |
| GSM8K | 11.3 | **40.1** | 35.3 | **61.6** |
| C-Eval | 37.2 | **45.2** | 55.3 | **63.8** |
| IFEval (Prompt Strict-Acc.) | 14.6 | **20.0** | 16.8 | **29.0** |

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen2,
  title={Qwen2 Technical Report},
  year={2024}
}
```



## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/LICENSE) (11.1 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/README.md) (3.5 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/config.json) (618.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/configuration.json) (48.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/generation_config.json) (170.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/merges.txt) (1.6 MB)

- [model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/model.safetensors) (2.9 GB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/tokenizer_config.json) (1.3 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-1.5B-Instruct/vocab.json) (2.6 MB)


[Back to Main](../../)