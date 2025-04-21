
# repllama-v1-7b-lora-passage
---


## README([From Huggingface](https://huggingface.co/castorini/repllama-v1-7b-lora-passage))




# RepLLaMA-7B-Passage

[Fine-Tuning LLaMA for Multi-Stage Text Retrieval](https://arxiv.org/abs/2310.08319).
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, Jimmy Lin, arXiv 2023

This model is fine-tuned from LLaMA-2-7B using LoRA and the embedding size is 4096.

## Training Data
The model is fine-tuned on the training split of [MS MARCO Passage Ranking](https://microsoft.github.io/msmarco/Datasets) datasets for 1 epoch.
Please check our paper for details.

## Usage

Below is an example to encode a query and a passage, and then compute their similarity using their embedding.

```python
import torch
from paddlenlp.transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig

def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = get_model('castorini/repllama-v1-7b-lora-passage')

# Define query and passage inputs
query = "What is llama?"
title = "Llama"
passage = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
query_input = tokenizer(f'query: {query}</s>', return_tensors='pt')
passage_input = tokenizer(f'passage: {title} {passage}</s>', return_tensors='pt')

# Run the model forward to compute embeddings and query-passage similarity score
with torch.no_grad():
    # compute query embedding
    query_outputs = model(**query_input)
    query_embedding = query_outputs.last_hidden_state[0][-1]
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

    # compute passage embedding
    passage_outputs = model(**passage_input)
    passage_embeddings = passage_outputs.last_hidden_state[0][-1]
    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=0)

    # compute similarity score
    score = torch.dot(query_embedding, passage_embeddings)
    print(score)

```
## Batch inference and training
An unofficial replication of the inference and training code can be found [here](https://github.com/texttron/tevatron/tree/main/examples/repllama)


## Citation

If you find our paper or models helpful, please consider cite as follows:

```
@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval}, 
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/README.md) (2.7 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/config.json) (1.3 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/generation_config.json) (167.0 B)

- [model_state-00001-of-00003.pdparams](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/model_state-00001-of-00003.pdparams) (9.2 GB)

- [model_state-00002-of-00003.pdparams](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/model_state-00002-of-00003.pdparams) (9.2 GB)

- [model_state-00003-of-00003.pdparams](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/model_state-00003-of-00003.pdparams) (6.7 GB)

- [model_state.pdparams.index.json](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/model_state.pdparams.index.json) (24.2 KB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/sentencepiece.bpe.model) (488.0 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/tokenizer.json) (1.8 MB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/tokenizer.model) (488.0 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/castorini/repllama-v1-7b-lora-passage/tokenizer_config.json) (745.0 B)


[Back to Main](../../)