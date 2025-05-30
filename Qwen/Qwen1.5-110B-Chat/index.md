
# Qwen1.5-110B-Chat
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen1.5-110B-Chat))



# Qwen1.5-110B-Chat


## Introduction

Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include: 

* 9 model sizes, including 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, and 110B dense models, and an MoE model of 14B with 2.7B activated;
* Significant performance improvement in human preference for chat models;
* Multilingual support of both base and chat models;
* Stable support of 32K context length for models of all sizes
* No need of `trust_remote_code`.

For more details, please refer to our [blog post](https://qwenlm.github.io/blog/qwen1.5/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).
<br>

## Model Details
Qwen1.5 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes. For the beta version, temporarily we did not include GQA (except for 32B and 110B) and the mixture of SWA and full attention.

## Training details
We pretrained the models with a large amount of data, and we post-trained the models with both supervised finetuning and direct preference optimization.


## Requirements
The code of Qwen1.5 has been in the latest Hugging face transformers and we advise you to install `transformers>=4.37.0`, or you might encounter the following error:
```
KeyError: 'qwen2'
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-110B-Chat",
    
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-110B-Chat")

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


## Tips

* If you encounter code switching or other bad cases, we advise you to use our provided hyper-parameters in `generation_config.json`.


## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```




## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/LICENSE) (6.7 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/README.md) (4.0 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/config.json) (620.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/configuration.json) (73.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/generation_config.json) (206.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/merges.txt) (1.6 MB)

- [model-00001-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00001-of-00062.safetensors) (3.4 GB)

- [model-00002-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00002-of-00062.safetensors) (3.3 GB)

- [model-00003-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00003-of-00062.safetensors) (3.6 GB)

- [model-00004-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00004-of-00062.safetensors) (3.3 GB)

- [model-00005-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00005-of-00062.safetensors) (3.3 GB)

- [model-00006-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00006-of-00062.safetensors) (3.6 GB)

- [model-00007-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00007-of-00062.safetensors) (3.3 GB)

- [model-00008-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00008-of-00062.safetensors) (3.3 GB)

- [model-00009-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00009-of-00062.safetensors) (3.6 GB)

- [model-00010-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00010-of-00062.safetensors) (3.3 GB)

- [model-00011-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00011-of-00062.safetensors) (3.3 GB)

- [model-00012-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00012-of-00062.safetensors) (3.6 GB)

- [model-00013-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00013-of-00062.safetensors) (3.3 GB)

- [model-00014-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00014-of-00062.safetensors) (3.3 GB)

- [model-00015-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00015-of-00062.safetensors) (3.6 GB)

- [model-00016-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00016-of-00062.safetensors) (3.3 GB)

- [model-00017-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00017-of-00062.safetensors) (3.3 GB)

- [model-00018-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00018-of-00062.safetensors) (3.6 GB)

- [model-00019-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00019-of-00062.safetensors) (3.3 GB)

- [model-00020-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00020-of-00062.safetensors) (3.3 GB)

- [model-00021-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00021-of-00062.safetensors) (3.6 GB)

- [model-00022-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00022-of-00062.safetensors) (3.3 GB)

- [model-00023-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00023-of-00062.safetensors) (3.3 GB)

- [model-00024-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00024-of-00062.safetensors) (3.6 GB)

- [model-00025-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00025-of-00062.safetensors) (3.3 GB)

- [model-00026-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00026-of-00062.safetensors) (3.3 GB)

- [model-00027-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00027-of-00062.safetensors) (3.6 GB)

- [model-00028-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00028-of-00062.safetensors) (3.3 GB)

- [model-00029-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00029-of-00062.safetensors) (3.3 GB)

- [model-00030-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00030-of-00062.safetensors) (3.6 GB)

- [model-00031-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00031-of-00062.safetensors) (3.3 GB)

- [model-00032-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00032-of-00062.safetensors) (3.3 GB)

- [model-00033-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00033-of-00062.safetensors) (3.6 GB)

- [model-00034-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00034-of-00062.safetensors) (3.3 GB)

- [model-00035-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00035-of-00062.safetensors) (3.3 GB)

- [model-00036-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00036-of-00062.safetensors) (3.6 GB)

- [model-00037-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00037-of-00062.safetensors) (3.3 GB)

- [model-00038-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00038-of-00062.safetensors) (3.3 GB)

- [model-00039-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00039-of-00062.safetensors) (3.6 GB)

- [model-00040-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00040-of-00062.safetensors) (3.3 GB)

- [model-00041-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00041-of-00062.safetensors) (3.3 GB)

- [model-00042-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00042-of-00062.safetensors) (3.6 GB)

- [model-00043-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00043-of-00062.safetensors) (3.3 GB)

- [model-00044-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00044-of-00062.safetensors) (3.3 GB)

- [model-00045-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00045-of-00062.safetensors) (3.6 GB)

- [model-00046-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00046-of-00062.safetensors) (3.3 GB)

- [model-00047-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00047-of-00062.safetensors) (3.3 GB)

- [model-00048-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00048-of-00062.safetensors) (3.6 GB)

- [model-00049-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00049-of-00062.safetensors) (3.3 GB)

- [model-00050-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00050-of-00062.safetensors) (3.3 GB)

- [model-00051-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00051-of-00062.safetensors) (3.6 GB)

- [model-00052-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00052-of-00062.safetensors) (3.3 GB)

- [model-00053-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00053-of-00062.safetensors) (3.3 GB)

- [model-00054-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00054-of-00062.safetensors) (3.6 GB)

- [model-00055-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00055-of-00062.safetensors) (3.3 GB)

- [model-00056-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00056-of-00062.safetensors) (3.3 GB)

- [model-00057-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00057-of-00062.safetensors) (3.6 GB)

- [model-00058-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00058-of-00062.safetensors) (3.3 GB)

- [model-00059-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00059-of-00062.safetensors) (3.3 GB)

- [model-00060-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00060-of-00062.safetensors) (3.6 GB)

- [model-00061-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00061-of-00062.safetensors) (2.3 GB)

- [model-00062-of-00062.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model-00062-of-00062.safetensors) (2.3 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/model.safetensors.index.json) (77.2 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/tokenizer_config.json) (1.3 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen1.5-110B-Chat/vocab.json) (2.6 MB)


[Back to Main](../../)