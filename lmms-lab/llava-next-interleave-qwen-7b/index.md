
# llava-next-interleave-qwen-7b
---


## README([From Huggingface](https://huggingface.co/lmms-lab/llava-next-interleave-qwen-7b))

---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{}
---

# LLaVA-Next Interleave Model Card
## Update
Note that we previous uploaded a wrong checkppoint. Now we have fixed it. Please download the new checkpoint.

## Model Details

Model type: LLaVA-Next Interleave is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture. 

Base LLM: Qwen/Qwen1.5-7B-Chat

### Model Description

**Repository:** https://github.com/LLaVA-VL/LLaVA-NeXT

**Primary intended uses:** The primary use of LLaVA-Next Interleave is research on large multimodal models and chatbots. This is only for research exploration, and prohibited for commercial usage.

**Primary intended users:** The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

### License Notices
  
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the OpenAI Terms of Use for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. Llama-1/2 community license for LLaMA-2 and Vicuna-v1.5, [Tongyi Qianwen LICENSE AGREEMENT](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT) and [META LLAMA 3 COMMUNITY LICENSE AGREEMENT](https://llama.meta.com/llama3/license/)). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.

## How to Get Started with the Model

Use the code below to get started with the model.

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
# install llava-next
...
# download the ckpt
...
bash playground/demo/interleave_demo.py --model_path path/to/ckpt
```



## Evaluation

Use the code below to evaluate the model.

Please first edit /path/to/ckpt to the path of checkpoint, /path/to/images to the path of "interleave_data" in scripts/interleave/eval_all.sh and then run
```bash
bash scripts/interleave/eval_all.sh
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/README.md) (2.4 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/added_tokens.json) (80.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/config.json) (1.7 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/generation_config.json) (294.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/merges.txt) (1.6 MB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/model_state.pdparams) (15.2 GB)

- [processor/eval/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/processor/eval/preprocessor_config.json) (368.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/tokenizer_config.json) (1.3 KB)

- [trainer_state.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/trainer_state.json) (694.5 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/lmms-lab/llava-next-interleave-qwen-7b/vocab.json) (2.6 MB)


[Back to Main](../../)