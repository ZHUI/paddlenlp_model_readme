
# Mixtral-8x7B-v0.1
---


## README([From Huggingface](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1))


# Model Card for Mixtral-8x7B
The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mistral-8x7B outperforms Llama 2 70B on most benchmarks we tested.

For full details of this model please read our [release blog post](https://mistral.ai/news/mixtral-of-experts/).

## Warning
This repo contains weights that are compatible with [vLLM](https://github.com/vllm-project/vllm) serving of the model as well as Hugging Face [transformers](https://github.com/huggingface/transformers) library. It is based on the original Mixtral [torrent release](magnet:?xt=urn:btih:5546272da9065eddeb6fcd7ffddeef5b75be79a7&dn=mixtral-8x7b-32kseqlen&tr=udp%3A%2F%http://2Fopentracker.i2p.rocks%3A6969%2Fannounce&tr=http%3A%2F%http://2Ftracker.openbittorrent.com%3A80%2Fannounce), but the file format and parameter names are different. Please note that model cannot (yet) be instantiated with HF.

## Run the model


```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pd")

outputs = model.generate(**inputs, max_new_tokens=20)[0]
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

By default, transformers will load the model in full precision. Therefore you might be interested to further reduce down the memory requirements to run the model through the optimizations we offer in HF ecosystem:

### In half-precision

Note `float16` precision only works on GPU devices

<details>
<summary> Click to expand </summary>

```diff
+ import paddle
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, dtype=paddle.float16).to(0)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pd").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)[0]
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

### Lower precision using (8-bit & 4-bit) using `bitsandbytes`

<details>
<summary> Click to expand </summary>

```diff
+ import paddle
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pd").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)[0]
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

### Load the model with Flash Attention 2

<details>
<summary> Click to expand </summary>

```diff
+ import paddle
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

+ model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=True)

text = "Hello my name is"
+ inputs = tokenizer(text, return_tensors="pd").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)[0]
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</details>

## Notice
Mixtral-8x7B is a pretrained base model and therefore does not have any moderation mechanisms.

# The Mistral AI Team
Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Louis Ternon, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/README.md) (4.0 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/config.json) (972.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/generation_config.json) (146.0 B)

- [model-00001-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00001-of-00019.safetensors) (4.6 GB)

- [model-00002-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00002-of-00019.safetensors) (4.6 GB)

- [model-00003-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00003-of-00019.safetensors) (4.6 GB)

- [model-00004-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00004-of-00019.safetensors) (4.6 GB)

- [model-00005-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00005-of-00019.safetensors) (4.6 GB)

- [model-00006-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00006-of-00019.safetensors) (4.6 GB)

- [model-00007-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00007-of-00019.safetensors) (4.6 GB)

- [model-00008-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00008-of-00019.safetensors) (4.6 GB)

- [model-00009-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00009-of-00019.safetensors) (4.6 GB)

- [model-00010-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00010-of-00019.safetensors) (4.6 GB)

- [model-00011-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00011-of-00019.safetensors) (4.6 GB)

- [model-00012-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00012-of-00019.safetensors) (4.6 GB)

- [model-00013-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00013-of-00019.safetensors) (4.6 GB)

- [model-00014-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00014-of-00019.safetensors) (4.6 GB)

- [model-00015-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00015-of-00019.safetensors) (4.6 GB)

- [model-00016-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00016-of-00019.safetensors) (4.6 GB)

- [model-00017-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00017-of-00019.safetensors) (4.6 GB)

- [model-00018-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00018-of-00019.safetensors) (4.6 GB)

- [model-00019-of-00019.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model-00019-of-00019.safetensors) (3.9 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/model.safetensors.index.json) (92.4 KB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/sentencepiece.bpe.model) (481.9 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/special_tokens_map.json) (72.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/tokenizer.json) (1.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x7B-v0.1/tokenizer_config.json) (967.0 B)


[Back to Main](../../)