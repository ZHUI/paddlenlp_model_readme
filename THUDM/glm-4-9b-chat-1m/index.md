
# glm-4-9b-chat-1m
---


## README([From Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m))



# GLM-4-9B-Chat-1M

Read this in [English](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m/file/view/master?fileName=README_en.md&status=1).

**2024/08/12, 本仓库代码已更新并使用 `transforemrs>=4.44.0`, 请及时更新依赖。**
**2024/07/24，我们发布了与长文本相关的最新技术解读，关注 [这里](https://medium.com/@ChatGLM/glm-long-scaling-pre-trained-model-contexts-to-millions-caa3c48dea85) 查看我们在训练 GLM-4-9B 开源模型中关于长文本技术的技术报告**

## 模型介绍

GLM-4-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源版本。
在语义、数学、推理、代码和知识等多方面的数据集测评中，GLM-4-9B 及其人类偏好对齐的版本 GLM-4-9B-Chat 均表现出较高的性能。
除了能进行多轮对话，GLM-4-9B-Chat 还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大 128K
上下文）等高级功能。
本代模型增加了多语言支持，支持包括日语，韩语，德语在内的 26 种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的模型。

## 评测结果

### 长文本

在 1M 的上下文长度下进行[大海捞针实验](https://github.com/LargeWorldModel/LWM/blob/main/scripts/eval_needle.py)，结果如下：

![![needle](https://raw.githubusercontent.com/THUDM/GLM-4/main/resources/eval_needle.jpeg)

在 LongBench-Chat 上对长文本能力进行了进一步评测，结果如下:

![![leaderboard](https://raw.githubusercontent.com/THUDM/GLM-4/main/resources/longbench.png)

**本仓库是 GLM-4-9B-Chat-1M 的模型仓库，支持`1M`上下文长度。**

## 运行模型

**更多推理代码和依赖信息，请访问我们的 [github](https://github.com/THUDM/GLM-4)。**

**请严格按照[依赖](https://github.com/THUDM/GLM-4/blob/main/basic_demo/requirements.txt)安装，否则无法正常运行。**

使用 transformers 后端进行推理:

```python
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/glm-4-9b-chat-1m",trust_remote_code=True)

query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pd",
                                       return_dict=True
                                       )

inputs = inputs
model = AutoModelForCausalLM.from_pretrained(
    "ZhipuAI/glm-4-9b-chat-1m",
    dtype=paddle.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

使用 VLLM后端进行推理:

```python
from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams
from modelscope import snapshot_download

# GLM-4-9B-Chat-1M
# max_model_len, tp_size = 1048576, 4

# GLM-4-9B-Chat
max_model_len, tp_size = 131072, 1
model_name = snapshot_download("ZhipuAI/glm-4-9b-chat-1m")
prompt = '你好'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
    enforce_eager=True,
    # GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
    # enable_chunked_prefill=True,
    # max_num_batched_tokens=8192
)
stop_token_ids = [151329, 151336, 151338]
sampling_params = SamplingParams(temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids)

inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True)[0]
outputs = llm.generate(prompt_token_ids=[inputs], sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

## 协议

GLM-4 模型的权重的使用则需要遵循 [LICENSE](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat/file/view/master?fileName=LICENSE&status=0)。


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

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/LICENSE) (6.4 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/README.md) (6.1 KB)

- [README_en.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/README_en.md) (6.6 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/config.json) (1.4 KB)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/configuration.json) (36.0 B)

- [configuration_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/configuration_chatglm.py) (2.2 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/generation_config.json) (173.0 B)

- [model-00001-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00001-of-00010.safetensors) (1.8 GB)

- [model-00002-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00002-of-00010.safetensors) (1.7 GB)

- [model-00003-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00003-of-00010.safetensors) (1.8 GB)

- [model-00004-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00004-of-00010.safetensors) (1.8 GB)

- [model-00005-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00005-of-00010.safetensors) (1.7 GB)

- [model-00006-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00006-of-00010.safetensors) (1.8 GB)

- [model-00007-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00007-of-00010.safetensors) (1.8 GB)

- [model-00008-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00008-of-00010.safetensors) (1.7 GB)

- [model-00009-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00009-of-00010.safetensors) (1.8 GB)

- [model-00010-of-00010.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model-00010-of-00010.safetensors) (1.5 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/model.safetensors.index.json) (28.1 KB)

- [modeling_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/modeling_chatglm.py) (56.8 KB)

- [tokenization_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/tokenization_chatglm.py) (15.6 KB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/tokenizer.model) (2.5 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-4-9b-chat-1m/tokenizer_config.json) (6.0 KB)


[Back to Main](../../)