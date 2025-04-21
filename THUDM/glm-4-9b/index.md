
# glm-4-9b
---


## README([From Huggingface](https://huggingface.co/THUDM/glm-4-9b))



# glm-4-9b

**2024/08/12, 本仓库代码已更新并使用 `transforemrs>=4.44.0`, 请及时更新依赖。**

GLM-4-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源版本。
在语义、数学、推理、代码和知识等多方面的数据集测评中，GLM-4-9B 及其人类偏好对齐的版本 GLM-4-9B-Chat 均表现出较高的性能。
除了能进行多轮对话，GLM-4-9B-Chat 还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大 128K
上下文）等高级功能。
本代模型增加了多语言支持，支持包括日语，韩语，德语在内的 26 种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的模型。

我们在一些典型任务上对 GLM-4-9B 基座模型进行了评测，结果如下：

| Model               |   MMLU   |  C-Eval  |   GPQA   |  GSM8K   |   MATH   | HumanEval |
|:--------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| Llama-3-8B          |   66.6   |   51.2   |    -     |   45.8   |    -     |     -     | 
| Llama-3-8B-Instruct |   68.4   |   51.3   |   34.2   |   79.6   |   30.0   |   62.2    |
| ChatGLM3-6B-Base    |   61.4   |   69.0   |    -     |   72.3   |   25.7   |     -     |
| GLM-4-9B            | **74.7** | **77.1** | **34.3** | **84.0** | **30.4** | **70.1**  |

更多推理代码和依赖信息，请访问我们的 [github](https://github.com/THUDM/GLM-4) 。

**本仓库是 GLM-4-9B 的基座版本，支持`8K`上下文长度。**

## 协议

GLM-4 模型的权重的使用则需要遵循 [LICENSE](LICENSE)。

## 引用

如果你觉得我们的工作有帮助的话，请考虑引用下列论文。

```
@misc{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools}, 
      author={Team GLM and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```




## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/LICENSE) (6.3 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/README.md) (3.4 KB)

- [README_en.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/README_en.md) (3.9 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/config.json) (1.3 KB)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/configuration.json) (36.0 B)

- [configuration_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/configuration_chatglm.py) (2.2 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/generation_config.json) (170.0 B)

- [model-00001-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00001-of-00010.safetensors) (1.8 GB)

- [model-00002-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00002-of-00010.safetensors) (1.7 GB)

- [model-00003-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00003-of-00010.safetensors) (1.8 GB)

- [model-00004-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00004-of-00010.safetensors) (1.8 GB)

- [model-00005-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00005-of-00010.safetensors) (1.7 GB)

- [model-00006-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00006-of-00010.safetensors) (1.8 GB)

- [model-00007-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00007-of-00010.safetensors) (1.8 GB)

- [model-00008-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00008-of-00010.safetensors) (1.7 GB)

- [model-00009-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00009-of-00010.safetensors) (1.8 GB)

- [model-00010-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model-00010-of-00010.safetensors) (1.5 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/model.safetensors.index.json) (28.1 KB)

- [modeling_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/modeling_chatglm.py) (46.3 KB)

- [tokenization_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/tokenization_chatglm.py) (15.3 KB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/tokenizer.model) (2.5 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b/tokenizer_config.json) (3.1 KB)


[Back to Main](../../)