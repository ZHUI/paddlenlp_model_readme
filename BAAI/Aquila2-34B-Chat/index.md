
# Aquila2-34B-Chat
---


## README([From Huggingface](https://huggingface.co/BAAI/Aquila2-34B-Chat))

---
desc: Aquila2-34B-Chat-PD 是基于 Aquila2-34B 模型优化得到的对话生成模型。它由 FlagAI 的 Aquila2 基础语言模型与飞桨（PaddlePaddle）NLP
  框架结合开发，经过权重转换与Infinity Instruct指令数据集二阶段监督微调后完成。
 
support_training: 0
 
tasks:
- 大语言模型
 
license: Apache License 2.0
 
dev_type:
- notebook

---





## <Aquila2-34B-Chat-PD>介绍

Aquila2-34B-Chat-PD 在模型结构与训练优化上取得了突破，是兼具灵活性与性能的高效对话生成解决方案。
a) 模型特点
•高效分词与长上下文支持
引入高压缩比的分词器，使模型能够高效处理文本，最大输入长度从 2048 tokens 扩展至 8192 tokens，特别适合长上下文处理场景。
•双阶段 SFT 优化
￮第一阶段：通过大规模通用数据集微调，提升模型的语言理解与生成能力。
￮第二阶段：基于对话数据集进一步优化，使模型能够更精准地进行上下文理解与逻辑生成。
•飞桨框架的深度支持
 Aquila2-34B 的权重通过飞桨 NLP 框架转换，使模型具备高效的训练与推理能力，便于用户在飞桨生态中  实现扩展与应用开发。
b) 技术优势
•增强的上下文能力：支持处理更长的对话或复杂任务，保证高质量输出。
•广泛适配性：经过双阶段优化，兼具通用语言能力与对话领域的深度适配性。
•生态开放性：飞桨框架的支持为模型的进一步优化与应用开发提供了便利。


### 如何使用
使用示例：
```
from modeling_aquila_pd import AquilaForCausalLM
from tokenizer_aquila_pd import AquilaTokenizer
from paddlenlp.transformers import AutoConfig

ckpt_path = "Aquila2_34B_Chat_PD"
config = AutoConfig.from_pretrained(ckpt_path)
tokenizer = AquilaTokenizer.from_pretrained(path)
model = AquilaForCausalLM.from_pretrained(path, config=config)

input_features = tokenizer("Hello, please introduce yourself.\n", return_tensors="pd")
outputs = model.generate(**input_features, max_new_tokens=128)
print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
```

### 训练数据介绍

Infinity Instruct 的训练数据包含了高质量、广覆盖的指令样本，专为提升模型的多任务学习和复杂场景适应能力而设计。通过以下两种核心数据集构建训练语料：

- **InfInstruct-7M**：由 744 万条指令构成，数据来源涵盖基础自然语言处理任务、逻辑推理、数学运算等多种领域，旨在帮助模型构建广泛的通用能力。
- **InfInstruct-Gen**：包含 145 万条高质量生成指令，专注于优化对话生成、多轮交流场景的表现。此数据集重点提升模型的生成能力、指令理解深度和任务执行的准确性。

为了确保训练数据的多样性与质量，Infinity Instruct 通过指令演化策略和标签系统对数据进行优化，不仅扩展了任务范围，还增强了数据的针对性。训练数据的设计过程强调以下几点：
1. **多领域覆盖**：确保数据集涵盖基础知识、专业技能和复杂推理任务。
2. **高难度指令**：包含挑战性任务，推动模型在极端和复杂任务中的性能提升。
3. **动态优化**：结合自动化诊断，持续生成和优化针对模型弱点的指令样本。

这些训练数据为模型提供了一个多层次、动态演进的学习框架，极大提升其在不同场景中的适应能力和性能。


### 数据评估及结果

| **评测集**         | **简介**                                                                 | **评测结果 (%)** |
|---------------------|--------------------------------------------------------------------------|------------------|
| **ARC-c**           | 测试常识推理能力，包含选择题形式的推理任务。                              | 72.88           |
| **ARC-e**           | 针对较复杂推理任务，包含更高难度的推理问题。                              | 85.89           |
| **Hellaswag**       | 测试常识推理和选择适当的推论能力。                                        | 61.04           |
| **MMLU**            | 包含多个领域的知识测试，评估广泛的学科知识。                              | 48.45           |
| **Winogrande**      | 测试模型在处理含有二义性句子的推理能力。                                  | 56.49           |
| **GSM8K**           | 评估数学推理能力，特别是解决多步骤数学问题。                              | 57.47           |
| **HumanEval**       | 用于评估代码生成能力，测试代码编写和理解。                                | 22.56           |
| **C-Eval**          | 针对中文常识推理和文本理解能力的评测集。                                  | 53.52           |
| **CMMLU**           | 综合多领域知识与推理能力的测试集。                                        | 53.72           |


## 相关论文以及引用信息
```
@article{InfinityInstruct2024,
  title={Infinity Instruct},
  author={Beijing Academy of Artificial Intelligence (BAAI)},
  journal={arXiv preprint arXiv:2406.XXXX},
  year={2024}
}
@article{zhao2024iidoptimizinginstructionlearning,
      title={Beyond IID: Optimizing Instruction Learning from the Perspective of Instruction Interaction and Dependency}, 
      author={Hanyu Zhao and Li Du and Yiming Ju and Chengwei Wu and Tengfei Pan},
      year={2024},
      eprint={2409.07045},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.07045},
}
@misc{zhang2024inifinitymath,
      title={InfinityMATH: A Scalable Instruction Tuning Dataset in Programmatic Mathematical Reasoning}, 
      author={Bo-Wen Zhang and Yan Yan and Lin Li and Guang Liu},
      year={2024},
      eprint={2408.07089},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.07089}, 
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/README.md) (5.9 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/config.json) (621.0 B)

- [configuration_aquila_pd.py](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/configuration_aquila_pd.py) (6.8 KB)

- [fusion_ops.py](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/fusion_ops.py) (9.6 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/generation_config.json) (76.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/merges.txt) (3.3 MB)

- [model-00001-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00001-of-00015.safetensors) (4.4 GB)

- [model-00002-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00002-of-00015.safetensors) (4.5 GB)

- [model-00003-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00003-of-00015.safetensors) (4.6 GB)

- [model-00004-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00004-of-00015.safetensors) (4.5 GB)

- [model-00005-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00005-of-00015.safetensors) (4.6 GB)

- [model-00006-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00006-of-00015.safetensors) (4.5 GB)

- [model-00007-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00007-of-00015.safetensors) (4.6 GB)

- [model-00008-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00008-of-00015.safetensors) (4.5 GB)

- [model-00009-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00009-of-00015.safetensors) (4.6 GB)

- [model-00010-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00010-of-00015.safetensors) (4.5 GB)

- [model-00011-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00011-of-00015.safetensors) (4.6 GB)

- [model-00012-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00012-of-00015.safetensors) (4.5 GB)

- [model-00013-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00013-of-00015.safetensors) (4.6 GB)

- [model-00014-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00014-of-00015.safetensors) (3.3 GB)

- [model-00015-of-00015.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model-00015-of-00015.safetensors) (1.6 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/model.safetensors.index.json) (44.2 KB)

- [modeling_aquila_pd.py](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/modeling_aquila_pd.py) (81.1 KB)

- [tokenizer_aquila_pd.py](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/tokenizer_aquila_pd.py) (13.4 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Aquila2-34B-Chat/vocab.json) (2.8 MB)


[Back to Main](../../)