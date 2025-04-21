
# QwQ-32B-Preview
---


## README([From Huggingface](https://huggingface.co/Qwen/QwQ-32B-Preview))



# QwQ-32B-Preview
<a href="https://chat.qwenlm.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5" style="display: inline-block; vertical-align: middle;"/>
</a>

## Introduction

**QwQ-32B-Preview** is an experimental research model developed by the Qwen Team, focused on advancing AI reasoning capabilities. As a preview release, it demonstrates promising analytical abilities while having several important limitations:

1. **Language Mixing and Code-Switching**: The model may mix languages or switch between them unexpectedly, affecting response clarity.
2. **Recursive Reasoning Loops**: The model may enter circular reasoning patterns, leading to lengthy responses without a conclusive answer.
3. **Safety and Ethical Considerations**: The model requires enhanced safety measures to ensure reliable and secure performance, and users should exercise caution when deploying it.
4. **Performance and Benchmark Limitations**: The model excels in math and coding but has room for improvement in other areas, such as common sense reasoning and nuanced language understanding.

**Specification**:
- Type: Causal Language Models
- Training Stage: Pretraining & Post-training
- Architecture: transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias
- Number of Parameters: 32.5B
- Number of Paramaters (Non-Embedding): 31.0B
- Number of Layers: 64
- Number of Attention Heads (GQA): 40 for Q and 8 for KV
- Context Length: Full 32,768 tokens

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwq-32b-preview/). You can also check Qwen2.5 [GitHub](https://github.com/QwenLM/Qwen2.5), and [Documentation](https://qwen.readthedocs.io/en/latest/).

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

model_name = "Qwen/QwQ-32B-Preview"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    
    
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r in strawberry."
messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pd").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{qwq-32b-preview,
    title = {QwQ: Reflect Deeply on the Boundaries of the Unknown},
    url = {https://qwenlm.github.io/blog/qwq-32b-preview/},
    author = {Qwen Team},
    month = {November},
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

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/LICENSE) (11.1 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/README.md) (4.5 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/config.json) (656.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/generation_config.json) (241.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/merges.txt) (1.6 MB)

- [model-00001-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00001-of-00017.safetensors) (3.6 GB)

- [model-00002-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00002-of-00017.safetensors) (3.6 GB)

- [model-00003-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00003-of-00017.safetensors) (3.6 GB)

- [model-00004-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00004-of-00017.safetensors) (3.6 GB)

- [model-00005-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00005-of-00017.safetensors) (3.6 GB)

- [model-00006-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00006-of-00017.safetensors) (3.6 GB)

- [model-00007-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00007-of-00017.safetensors) (3.6 GB)

- [model-00008-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00008-of-00017.safetensors) (3.6 GB)

- [model-00009-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00009-of-00017.safetensors) (3.6 GB)

- [model-00010-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00010-of-00017.safetensors) (3.6 GB)

- [model-00011-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00011-of-00017.safetensors) (3.6 GB)

- [model-00012-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00012-of-00017.safetensors) (3.6 GB)

- [model-00013-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00013-of-00017.safetensors) (3.6 GB)

- [model-00014-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00014-of-00017.safetensors) (3.6 GB)

- [model-00015-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00015-of-00017.safetensors) (3.6 GB)

- [model-00016-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00016-of-00017.safetensors) (3.6 GB)

- [model-00017-of-00017.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model-00017-of-00017.safetensors) (2.9 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/model.safetensors.index.json) (61.8 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/tokenizer_config.json) (7.2 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QwQ-32B-Preview/vocab.json) (2.6 MB)


[Back to Main](../../)