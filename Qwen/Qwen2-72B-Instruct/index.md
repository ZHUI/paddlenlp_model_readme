
# Qwen2-72B-Instruct
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2-72B-Instruct))



# Qwen2-72B-Instruct

## Introduction

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This repo contains the instruction-tuned 72B Qwen2 model.

Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics, reasoning, etc.

Qwen2-72B-Instruct supports a context length of up to 131,072 tokens, enabling the processing of extensive inputs. Please refer to [this section](#processing-long-texts) for detailed instructions on how to deploy Qwen2 for handling long texts.

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
    "Qwen/Qwen2-72B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-72B-Instruct")

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
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Processing Long Texts

To handle extensive inputs exceeding 32,768 tokens, we utilize [YARN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

For deployment, we recommend using vLLM. You can enable the long-context capabilities by following these steps:

1. **Install vLLM**: You can install vLLM by running the following command.

```bash
pip install "vllm>=0.4.3"
```

Or you can install vLLM from [source](https://github.com/vllm-project/vllm/).

2. **Configure Model Settings**: After downloading the model weights, modify the `config.json` file by including the below snippet:
    ```json
        {
            "architectures": [
                "Qwen2ForCausalLM"
            ],
            // ...
            "vocab_size": 152064,

            // adding the following snippets
            "rope_scaling": {
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "type": "yarn"
            }
        }
    ```
    This snippet enable YARN to support longer contexts.

3. **Model Deployment**: Utilize vLLM to deploy your model. For instance, you can set up an openAI-like server using the command:

    ```bash
    python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-72B-Instruct --model path/to/weights
    ```

    Then you can access the Chat API by:

    ```bash
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "Qwen2-72B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Your Long Input Here."}
        ]
        }'
    ```

    For further usage instructions of vLLM, please refer to our [Github](https://github.com/QwenLM/Qwen2).

**Note**: Presently, vLLM only supports static YARN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts**. We advise adding the `rope_scaling` configuration only when processing long contexts is required.

## Evaluation

We briefly compare Qwen2-72B-Instruct with similar-sized instruction-tuned LLMs, including our previous Qwen1.5-72B-Chat. The results are shown as follows:

| Datasets | Llama-3-70B-Instruct | Qwen1.5-72B-Chat | **Qwen2-72B-Instruct** |
| :--- | :---: | :---: | :---: |
| _**English**_ |  |  |  |
| MMLU | 82.0 | 75.6 | **82.3** |
| MMLU-Pro | 56.2 | 51.7 | **64.4** |
| GPQA | 41.9 | 39.4 | **42.4** |
| TheroemQA | 42.5 | 28.8 | **44.4** |
| MT-Bench | 8.95 | 8.61 | **9.12** |
| Arena-Hard | 41.1 | 36.1 | **48.1** |
| IFEval (Prompt Strict-Acc.) | 77.3 | 55.8 | **77.6** |
| _**Coding**_ |  |  |  |
| HumanEval | 81.7 | 71.3 | **86.0** |
| MBPP | **82.3** | 71.9 | 80.2 |
| MultiPL-E | 63.4 | 48.1 | **69.2** |
| EvalPlus | 75.2 | 66.9 | **79.0** |
| LiveCodeBench | 29.3 | 17.9 | **35.7** |
| _**Mathematics**_ |  |  |  |
| GSM8K | **93.0** | 82.7 | 91.1 |
| MATH | 50.4 | 42.5 | **59.7** |
| _**Chinese**_ |  |  |  |
| C-Eval | 61.6 | 76.1 | **83.8** |
| AlignBench | 7.42 | 7.28 | **8.27** |

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen2,
  title={Qwen2 Technical Report},
  year={2024}
}
```



## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/LICENSE) (6.7 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/README.md) (6.5 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/config.json) (621.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/configuration.json) (48.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/generation_config.json) (206.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/merges.txt) (1.6 MB)

- [model-00001-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00001-of-00037.safetensors) (3.5 GB)

- [model-00002-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00002-of-00037.safetensors) (3.7 GB)

- [model-00003-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00003-of-00037.safetensors) (3.6 GB)

- [model-00004-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00004-of-00037.safetensors) (3.7 GB)

- [model-00005-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00005-of-00037.safetensors) (3.7 GB)

- [model-00006-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00006-of-00037.safetensors) (3.7 GB)

- [model-00007-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00007-of-00037.safetensors) (3.6 GB)

- [model-00008-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00008-of-00037.safetensors) (3.7 GB)

- [model-00009-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00009-of-00037.safetensors) (3.7 GB)

- [model-00010-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00010-of-00037.safetensors) (3.7 GB)

- [model-00011-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00011-of-00037.safetensors) (3.6 GB)

- [model-00012-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00012-of-00037.safetensors) (3.7 GB)

- [model-00013-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00013-of-00037.safetensors) (3.7 GB)

- [model-00014-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00014-of-00037.safetensors) (3.7 GB)

- [model-00015-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00015-of-00037.safetensors) (3.6 GB)

- [model-00016-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00016-of-00037.safetensors) (3.7 GB)

- [model-00017-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00017-of-00037.safetensors) (3.7 GB)

- [model-00018-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00018-of-00037.safetensors) (3.7 GB)

- [model-00019-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00019-of-00037.safetensors) (3.6 GB)

- [model-00020-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00020-of-00037.safetensors) (3.7 GB)

- [model-00021-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00021-of-00037.safetensors) (3.7 GB)

- [model-00022-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00022-of-00037.safetensors) (3.7 GB)

- [model-00023-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00023-of-00037.safetensors) (3.6 GB)

- [model-00024-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00024-of-00037.safetensors) (3.7 GB)

- [model-00025-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00025-of-00037.safetensors) (3.7 GB)

- [model-00026-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00026-of-00037.safetensors) (3.7 GB)

- [model-00027-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00027-of-00037.safetensors) (3.6 GB)

- [model-00028-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00028-of-00037.safetensors) (3.7 GB)

- [model-00029-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00029-of-00037.safetensors) (3.7 GB)

- [model-00030-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00030-of-00037.safetensors) (3.7 GB)

- [model-00031-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00031-of-00037.safetensors) (3.6 GB)

- [model-00032-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00032-of-00037.safetensors) (3.7 GB)

- [model-00033-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00033-of-00037.safetensors) (3.7 GB)

- [model-00034-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00034-of-00037.safetensors) (3.7 GB)

- [model-00035-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00035-of-00037.safetensors) (3.6 GB)

- [model-00036-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00036-of-00037.safetensors) (3.7 GB)

- [model-00037-of-00037.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model-00037-of-00037.safetensors) (3.2 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/model.safetensors.index.json) (77.2 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/tokenizer_config.json) (1.3 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2-72B-Instruct/vocab.json) (2.6 MB)


[Back to Main](../../)