
# Qwen2.5-Coder-0.5B-Instruct
---


## README([From Huggingface](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct))




# Qwen2.5-Coder-0.5B-Instruct

## Introduction

Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). As of now, Qwen2.5-Coder has covered six mainstream model sizes, 0.5, 1.5, 3, 7, 14, 32 billion parameters, to meet the needs of different developers. Qwen2.5-Coder brings the following improvements upon CodeQwen1.5:

- Significantly improvements in **code generation**, **code reasoning** and **code fixing**. Base on the strong Qwen2.5, we scale up the training tokens into 5.5 trillion including source code, text-code grounding, Synthetic data, etc. Qwen2.5-Coder-32B has become the current state-of-the-art open-source codeLLM, with its coding abilities matching those of GPT-4o.
- A more comprehensive foundation for real-world applications such as **Code Agents**. Not only enhancing coding capabilities but also maintaining its strengths in mathematics and general competencies.

**This repo contains the instruction-tuned 0.5B Qwen2.5-Coder model**, which has the following features:
- Type: Causal Language Models
- Training Stage: Pretraining & Post-training
- Architecture: transformers with RoPE, SwiGLU, RMSNorm, Attention QKV bias and tied word embeddings
- Number of Parameters: 0.49B
- Number of Paramaters (Non-Embedding): 0.36B
- Number of Layers: 24
- Number of Attention Heads (GQA): 14 for Q and 2 for KV
- Context Length: Full 32,768 tokens
  
For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/), [GitHub](https://github.com/QwenLM/Qwen2.5-Coder), [Documentation](https://qwen.readthedocs.io/en/latest/), [Arxiv](https://arxiv.org/abs/2409.12186).

## Requirements

The code of Qwen2.5-Coder has been in the latest Hugging face `transformers` and we advise you to use the latest version of `transformers`.

With `transformers<4.37.0`, you will encounter the following error:
```
KeyError: 'qwen2'
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "write a quick sort algorithm."
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
```


## Evaluation & Performance

Detailed evaluation results are reported in this [📑 blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/).

For requirements on GPU memory and the respective throughput, see results [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{hui2024qwen2,
      title={Qwen2. 5-Coder Technical Report},
      author={Hui, Binyuan and Yang, Jian and Cui, Zeyu and Yang, Jiaxi and Liu, Dayiheng and Zhang, Lei and Liu, Tianyu and Zhang, Jiajun and Yu, Bowen and Dang, Kai and others},
      journal={arXiv preprint arXiv:2409.12186},
      year={2024}
}
@article{qwen2,
      title={Qwen2 Technical Report}, 
      author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
      journal={arXiv preprint arXiv:2407.10671},
      year={2024}
}
```




## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/LICENSE) (11.1 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/README.md) (4.9 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/config.json) (653.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/generation_config.json) (243.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/merges.txt) (1.6 MB)

- [model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors) (942.3 MB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/tokenizer_config.json) (7.1 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/Qwen2.5-Coder-0.5B-Instruct/vocab.json) (2.6 MB)


[Back to Main](../../)