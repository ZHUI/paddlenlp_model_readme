
# LLARA-passage
---


## README([From Huggingface](https://huggingface.co/BAAI/LLARA-passage))

---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
license: mit
---

For more details please refer to our github repo: https://github.com/FlagOpen/FlagEmbedding

# LLARA ([paper](https://arxiv.org/pdf/2312.15503))

In this project, we introduce LLaRA:
- EBAE: Embedding-Based Auto-Encoding.
- EBAR: Embedding-Based Auto-Regression. 


## Usage

```
import torch
from transformers import AutoModel, AutoTokenizer, LlamaModel

def get_query_inputs(queries, tokenizer, max_length=512):
    prefix = '"'
    suffix = '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>'
    prefix_ids = tokenizer(prefix, return_tensors=None)['input_ids']
    suffix_ids = tokenizer(suffix, return_tensors=None)['input_ids'][1:]
    queries_inputs = []
    for query in queries:
        inputs = tokenizer(query,
                           return_tensors=None,
                           max_length=max_length,
                           truncation=True,
                           add_special_tokens=False)
        inputs['input_ids'] = prefix_ids + inputs['input_ids'] + suffix_ids
        inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        queries_inputs.append(inputs)
    return tokenizer.pad(
            queries_inputs,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

def get_passage_inputs(passages, tokenizer, max_length=512):
    prefix = '"'
    suffix = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
    prefix_ids = tokenizer(prefix, return_tensors=None)['input_ids']
    suffix_ids = tokenizer(suffix, return_tensors=None)['input_ids'][1:]
    passages_inputs = []
    for passage in passages:
        inputs = tokenizer(passage,
                           return_tensors=None,
                           max_length=max_length,
                           truncation=True,
                           add_special_tokens=False)
        inputs['input_ids'] = prefix_ids + inputs['input_ids'] + suffix_ids
        inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        passages_inputs.append(inputs)
    return tokenizer.pad(
            passages_inputs,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('BAAI/LLARA-passage')
model = AutoModel.from_pretrained('BAAI/LLARA-passage')

# Define query and passage inputs
query = "What is llama?"
title = "Llama"
passage = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
query_input = get_query_inputs([query], tokenizer)
passage_input = get_passage_inputs([passage], tokenizer)


with torch.no_grad():
    # compute query embedding
    query_outputs = model(**query_input, return_dict=True, output_hidden_states=True)
    query_embedding = query_outputs.hidden_states[-1][:, -8:, :]
    query_embedding = torch.mean(query_embedding, dim=1)
    query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)

    # compute passage embedding
    passage_outputs = model(**passage_input, return_dict=True, output_hidden_states=True)
    passage_embeddings = passage_outputs.hidden_states[-1][:, -8:, :]
    passage_embeddings = torch.mean(passage_embeddings, dim=1)
    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, dim=-1)

    # compute similarity score
    score = query_embedding @ passage_embeddings.T
    print(score)

```


## Acknowledgement

Thanks to the authors of open-sourced datasets, including MSMARCO, BEIR, etc. 
Thanks to the open-sourced libraries like [Pyserini](https://github.com/castorini/pyserini).



## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
@misc{li2023making,
      title={Making Large Language Models A Better Foundation For Dense Retrieval}, 
      author={Chaofan Li and Zheng Liu and Shitao Xiao and Yingxia Shao},
      year={2023},
      eprint={2312.15503},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/README.md) (4.2 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/added_tokens.json) (247.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/config.json) (1.0 KB)

- [model_state-00001-of-00006.pdparams](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/model_state-00001-of-00006.pdparams) (4.5 GB)

- [model_state-00002-of-00006.pdparams](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/model_state-00002-of-00006.pdparams) (4.5 GB)

- [model_state-00003-of-00006.pdparams](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/model_state-00003-of-00006.pdparams) (4.5 GB)

- [model_state-00004-of-00006.pdparams](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/model_state-00004-of-00006.pdparams) (4.5 GB)

- [model_state-00005-of-00006.pdparams](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/model_state-00005-of-00006.pdparams) (4.5 GB)

- [model_state-00006-of-00006.pdparams](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/model_state-00006-of-00006.pdparams) (2.0 GB)

- [model_state.pdparams.index.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/model_state.pdparams.index.json) (22.5 KB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/sentencepiece.bpe.model) (488.0 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/special_tokens_map.json) (575.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/LLARA-passage/tokenizer_config.json) (3.5 KB)


[Back to Main](../../)