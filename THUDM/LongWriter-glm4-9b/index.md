
# LongWriter-glm4-9b
---


## README([From Huggingface](https://huggingface.co/THUDM/LongWriter-glm4-9b))

---
language:
- en
- zh
library_name: transformers
tags:
- Long Context
- chatglm
- llama
datasets:
  train:
    - AI-ModelScope/LongWriter-6k
pipeline_tag: text-generation
studios:
- ZhipuAI/LongWriter-glm4-9b-demo
---
# LongWriter-glm4-9b

<p align="center">
  🤖 <a href="https://modelscope.cn/datasets/ZhipuAI/LongWriter-6k" target="_blank">[LongWriter Dataset] </a> • 💻 <a href="https://github.com/THUDM/LongWriter" target="_blank">[Github Repo]</a> • 📃 <a href="https://arxiv.org/abs/2408.07055" target="_blank">[LongWriter Paper]</a> 
</p>

LongWriter-glm4-9b is trained based on [glm-4-9b](https://huggingface.co/THUDM/glm-4-9b), and is capable of generating 10,000+ words at once.


A simple demo for deployment of the model:
```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
import paddle
tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/LongWriter-glm4-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("ZhipuAI/LongWriter-glm4-9b", dtype=paddle.bfloat16, trust_remote_code=True, )
model = model.eval()
query = "Write a `10000`-word China travel guide"
response, history = model.chat(tokenizer, query, history=[], max_new_tokens=1024, temperature=0.5)
print(response)
```
Environment: `transformers==4.43.0`

License: [glm-4-9b License](https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/LICENSE)

## Citation

If you find our work useful, please consider citing LongWriter:

```
@article{bai2024longwriter,
  title={LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs}, 
  author={Yushi Bai and Jiajie Zhang and Xin Lv and Linzhi Zheng and Siqi Zhu and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  journal={arXiv preprint arXiv:2408.07055},
  year={2024}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/README.md) (1.7 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/config.json) (1.4 KB)

- [configuration_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/configuration_chatglm.py) (2.2 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/generation_config.json) (120.0 B)

- [model-00001-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/model-00001-of-00004.safetensors) (4.6 GB)

- [model-00002-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/model-00002-of-00004.safetensors) (4.6 GB)

- [model-00003-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/model-00003-of-00004.safetensors) (4.6 GB)

- [model-00004-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/model-00004-of-00004.safetensors) (3.7 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/model.safetensors.index.json) (28.1 KB)

- [modeling_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/modeling_chatglm.py) (43.4 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/special_tokens_map.json) (35.0 B)

- [tokenization_chatglm.py](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/tokenization_chatglm.py) (10.7 KB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/tokenizer.model) (2.5 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/LongWriter-glm4-9b/tokenizer_config.json) (843.0 B)


[Back to Main](../../)