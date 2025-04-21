
# Qwen2.5-14B-Instruct
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct))



# Qwen2.5-14B-Instruct

## Introduction

Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters. Qwen2.5 brings the following improvements upon Qwen2:

- Significantly **more knowledge** and has greatly improved capabilities in **coding** and **mathematics**, thanks to our specialized expert models in these domains.
- Significant improvements in **instruction following**, **generating long texts** (over 8K tokens), **understanding structured data** (e.g, tables), and **generating structured outputs** especially JSON. **More resilient to the diversity of system prompts**, enhancing role-play implementation and condition-setting for chatbots.
- **Long-context Support** up to 128K tokens and can generate up to 8K tokens.
- **Multilingual support** for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more. 

**This repo contains the instruction-tuned 14B Qwen2.5 model**, which has the following features:
- Type: Causal Language Models
- Training Stage: Pretraining & Post-training
- Architecture: transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias
- Number of Parameters: 14.7B
- Number of Paramaters (Non-Embedding): 13.1B
- Number of Layers: 48
- Number of Attention Heads (GQA): 40 for Q and 8 for KV
- Context Length: Full 131,072 tokens and generation 8192 tokens
  - Please refer to [this section](#processing-long-texts) for detailed instructions on how to deploy Qwen2.5 for handling long texts.

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2.5/), [GitHub](https://github.com/QwenLM/Qwen2.5), and [Documentation](https://qwen.readthedocs.io/en/latest/).

## Requirements

The code of Qwen2.5 has been in the latest Hugging face `transformers` and we advise you to use the latest version of `transformers`.

With `transformers<4.37.0`, you will encounter the following error:
```
KeyError: 'qwen2'
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "qwen/Qwen2.5-14B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pd")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)[0]
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Processing Long Texts

The current `config.json` is set for context length up to 32,768 tokens.
To handle extensive inputs exceeding 32,768 tokens, we utilize [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

For supported frameworks, you could add the following to `config.json` to enable YaRN:
```json
{
  ...,
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
```

For deployment, we recommend using vLLM. 
Please refer to our [Documentation](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) for usage if you are not familar with vLLM.
Presently, vLLM only supports static YARN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts**. 
We advise adding the `rope_scaling` configuration only when processing long contexts is required.

## Evaluation & Performance

Detailed evaluation results are reported in this [📑 blog](https://qwenlm.github.io/blog/qwen2.5/).

For requirements on GPU memory and the respective throughput, see results [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{qwen2.5,
    title = {Qwen2.5: A Party of Foundation Models},
    url = {https://qwenlm.github.io/blog/qwen2.5/},
    author = {Qwen Team},
    month = {September},
    year = {2024}
}

@article{qwen2,
      title={Qwen2 Technical Report}, 
      author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
      journal={arXiv preprint arXiv:2407.10671},
      year={2024}
}
```



## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/LICENSE) (11.1 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/README.md) (5.9 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/config.json) (657.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/configuration.json) (2.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/generation_config.json) (242.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/merges.txt) (1.6 MB)

- [model-00001-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model-00001-of-00008.safetensors) (3.6 GB)

- [model-00002-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model-00002-of-00008.safetensors) (3.7 GB)

- [model-00003-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model-00003-of-00008.safetensors) (3.7 GB)

- [model-00004-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model-00004-of-00008.safetensors) (3.7 GB)

- [model-00005-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model-00005-of-00008.safetensors) (3.7 GB)

- [model-00006-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model-00006-of-00008.safetensors) (3.7 GB)

- [model-00007-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model-00007-of-00008.safetensors) (3.7 GB)

- [model-00008-of-00008.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model-00008-of-00008.safetensors) (1.6 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/model.safetensors.index.json) (46.4 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/tokenizer_config.json) (7.1 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-14B-Instruct/vocab.json) (2.6 MB)


[Back to Main](../../)