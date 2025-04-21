
# vicuna-7b-v1.5
---


## README([From Huggingface](https://huggingface.co/paddlemix/llava/vicuna-7b-v1.5))

---
inference: false
license: llama2
---

# Vicuna Model Card

## Model Details

Vicuna is a chat assistant trained by fine-tuning Llama 2 on user-shared conversations collected from ShareGPT.

- **Developed by:** [LMSYS](https://lmsys.org/)
- **Model type:** An auto-regressive language model based on the transformer architecture
- **License:** Llama 2 Community License Agreement	
- **Finetuned from model:** [Llama 2](https://arxiv.org/abs/2307.09288)

### Model Sources

- **Repository:** https://github.com/lm-sys/FastChat
- **Blog:** https://lmsys.org/blog/2023-03-30-vicuna/
- **Paper:** https://arxiv.org/abs/2306.05685
- **Demo:** https://chat.lmsys.org/

## Uses

The primary use of Vicuna is research on large language models and chatbots.
The primary intended users of the model are researchers and hobbyists in natural language processing, machine learning, and artificial intelligence.

## How to Get Started with the Model

- Command line interface: https://github.com/lm-sys/FastChat#vicuna-weights
- APIs (OpenAI API, Huggingface API): https://github.com/lm-sys/FastChat/tree/main#api  

## Training Details

Vicuna v1.5 is fine-tuned from Llama 2 with supervised instruction fine-tuning.
The training data is around 125K conversations collected from ShareGPT.com.
See more details in the "Training Details of Vicuna Models" section in the appendix of this [paper](https://arxiv.org/pdf/2306.05685.pdf).

## Evaluation

![![Evaluation Results](https://github.com/lm-sys/lm-sys.github.io/blob/main/public/images/webdata/vicuna_v1.5_eval.png?raw=true)

Vicuna is evaluated with standard benchmarks, human preference, and LLM-as-a-judge. See more details in this [paper](https://arxiv.org/pdf/2306.05685.pdf) and [leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard).

## Difference between different versions of Vicuna

See [vicuna_weights_version.md](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md)



## Model Files

- [.gitattributes](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/.gitattributes) (1.5 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/README.md) (1.9 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/config.json) (1.2 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/generation_config.json) (162.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/model_state.pdparams) (12.6 GB)

- [processor/eval/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/processor/eval/preprocessor_config.json) (584.0 B)

- [processor/train/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/processor/train/preprocessor_config.json) (584.0 B)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/sentencepiece.bpe.model) (488.0 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/special_tokens_map.json) (438.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/llava/vicuna-7b-v1.5/tokenizer_config.json) (749.0 B)


[Back to Main](../../../)