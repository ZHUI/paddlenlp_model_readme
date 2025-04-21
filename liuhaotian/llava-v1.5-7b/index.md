
# llava-v1.5-7b
---


## README([From Huggingface](https://huggingface.co/liuhaotian/llava-v1.5-7b))

---
inference: false
pipeline_tag: image-text-to-text
---

<br>
<br>

# LLaVA Model Card

## Model details

**Model type:**
LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal instruction-following data.
It is an auto-regressive language model, based on the transformer architecture.

**Model date:**
LLaVA-v1.5-7B was trained in September 2023.

**Paper or resources for more information:**
https://llava-vl.github.io/

## License
Llama 2 is licensed under the LLAMA 2 Community License, 
Copyright (c) Meta Platforms, Inc. All Rights Reserved.

**Where to send questions or comments about the model:**
https://github.com/haotian-liu/LLaVA/issues

## Intended use
**Primary intended uses:**
The primary use of LLaVA is research on large multimodal models and chatbots.

**Primary intended users:**
The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

## Training dataset
- 558K filtered image-text pairs from LAION/CC/SBU, captioned by BLIP.
- 158K GPT-generated multimodal instruction-following data.
- 450K academic-task-oriented VQA data mixture.
- 40K ShareGPT data.

## Evaluation dataset
A collection of 12 benchmarks, including 5 academic VQA benchmarks and 7 recent benchmarks specifically proposed for instruction-following LMMs.



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/README.md) (1.4 KB)

- [chat_template.json](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/chat_template.json) (60.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/config.json) (1.1 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/generation_config.json) (125.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/model_state.pdparams) (25.2 GB)

- [processor/eval/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/processor/eval/preprocessor_config.json) (380.0 B)

- [processor/train/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/processor/train/preprocessor_config.json) (380.0 B)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/sentencepiece.bpe.model) (488.0 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/special_tokens_map.json) (438.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/liuhaotian/llava-v1.5-7b/tokenizer_config.json) (749.0 B)


[Back to Main](../../)