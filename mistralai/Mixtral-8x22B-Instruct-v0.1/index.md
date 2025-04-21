
# Mixtral-8x22B-Instruct-v0.1
---


## README([From Huggingface](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1))



# Model Card for Mixtral-8x22B-Instruct-v0.1
The Mixtral-8x22B-Instruct-v0.1 Large Language Model (LLM) is an instruct fine-tuned version of the [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1).

## Run the model 
```python
from modelscope import AutoModelForCausalLM
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import (
    Tool,
    Function,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.normalize import ChatCompletionRequest

device = "cuda" # the device to load the model onto

tokenizer_v3 = MistralTokenizer.v3()

mistral_query = ChatCompletionRequest(
    tools=[
        Tool(
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            )
        )
    ],
    messages=[
        UserMessage(content="What's the weather like today in Paris"),
    ],
    model="test",
)

encodeds = tokenizer_v3.encode_chat_completion(mistral_query).tokens
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-Instruct-v0.1")
model_inputs = encodeds
model

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
sp_tokenizer = tokenizer_v3.instruct_tokenizer.tokenizer
decoded = sp_tokenizer.decode(generated_ids[0])
print(decoded)
```
Alternatively, you can run this example with the Hugging Face tokenizer.
To use this example, you'll need transformers version 4.39.0 or higher.
```console
pip install transformers==4.39.0
```
```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
conversation=[
    {"role": "user", "content": "What's the weather like in Paris?"},
    {
        "role": "tool_calls",
        "content": [
            {
                "name": "get_current_weather",
                "arguments": {"location": "Paris, France", "format": "celsius"},
                
            }
        ]
    },
    {
        "role": "tool_results",
        "content": {"content": 22}
    },
    {"role": "assistant", "content": "The current temperature in Paris, France is 22 degrees Celsius."},
    {"role": "user", "content": "What about San Francisco?"}
]


tools = [{"type": "function", "function": {"name":"get_current_weather", "description": "Get▁the▁current▁weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "format": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit to use. Infer this from the users location."}},"required":["location","format"]}}}]

# render the tool use prompt as a string:
tool_use_prompt = tokenizer.apply_chat_template(
            conversation,
            chat_template="tool_use",
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,

)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-Instruct-v0.1")

inputs = tokenizer(tool_use_prompt, return_tensors="pd")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

# Instruct tokenizer
The HuggingFace tokenizer included in this release should match our own. To compare: 
`pip install mistral-common`

```py
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.normalize import ChatCompletionRequest

from paddlenlp.transformers import AutoTokenizer

tokenizer_v3 = MistralTokenizer.v3()

mistral_query = ChatCompletionRequest(
    messages=[
        UserMessage(content="How many experts ?"),
        AssistantMessage(content="8"),
        UserMessage(content="How big ?"),
        AssistantMessage(content="22B"),
        UserMessage(content="Noice 🎉 !"),
    ],
    model="test",
)
hf_messages = mistral_query.model_dump()['messages']

tokenized_mistral = tokenizer_v3.encode_chat_completion(mistral_query).tokens

tokenizer_hf = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x22B-Instruct-v0.1')
tokenized_hf = tokenizer_hf.apply_chat_template(hf_messages, tokenize=True)

assert tokenized_hf == tokenized_mistral
```

# Function calling and special tokens
This tokenizer includes more special tokens, related to function calling : 
- [TOOL_CALLS]
- [AVAILABLE_TOOLS]
- [/AVAILABLE_TOOLS]
- [TOOL_RESULTS]
- [/TOOL_RESULTS]

If you want to use this model with function calling, please be sure to apply it similarly to what is done in our [SentencePieceTokenizerV3](https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/sentencepiece.py#L299).

# The Mistral AI Team
Albert Jiang, Alexandre Sablayrolles, Alexis Tacnet, Antoine Roux,
Arthur Mensch, Audrey Herblin-Stoop, Baptiste Bout, Baudouin de Monicault,
Blanche Savary, Bam4d, Caroline Feldman, Devendra Singh Chaplot,
Diego de las Casas, Eleonore Arcelin, Emma Bou Hanna, Etienne Metzger,
Gianna Lengyel, Guillaume Bour, Guillaume Lample, Harizo Rajaona,
Jean-Malo Delignon, Jia Li, Justus Murke, Louis Martin, Louis Ternon,
Lucile Saulnier, Lélio Renard Lavaud, Margaret Jennings, Marie Pellat,
Marie Torelli, Marie-Anne Lachaux, Nicolas Schuhl, Patrick von Platen,
Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao,
Thibaut Lavril, Timothée Lacroix, Théophile Gervet, Thomas Wang,
Valera Nemychnikova, William El Sayed, William Marshall



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/README.md) (6.3 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/config.json) (710.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/configuration.json) (73.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/generation_config.json) (116.0 B)

- [model-00001-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00001-of-00059.safetensors) (4.5 GB)

- [model-00002-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00002-of-00059.safetensors) (4.5 GB)

- [model-00003-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00003-of-00059.safetensors) (4.5 GB)

- [model-00004-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00004-of-00059.safetensors) (4.5 GB)

- [model-00005-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00005-of-00059.safetensors) (4.5 GB)

- [model-00006-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00006-of-00059.safetensors) (4.5 GB)

- [model-00007-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00007-of-00059.safetensors) (4.5 GB)

- [model-00008-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00008-of-00059.safetensors) (4.5 GB)

- [model-00009-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00009-of-00059.safetensors) (4.5 GB)

- [model-00010-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00010-of-00059.safetensors) (4.5 GB)

- [model-00011-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00011-of-00059.safetensors) (4.5 GB)

- [model-00012-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00012-of-00059.safetensors) (4.5 GB)

- [model-00013-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00013-of-00059.safetensors) (4.5 GB)

- [model-00014-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00014-of-00059.safetensors) (4.5 GB)

- [model-00015-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00015-of-00059.safetensors) (4.5 GB)

- [model-00016-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00016-of-00059.safetensors) (4.5 GB)

- [model-00017-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00017-of-00059.safetensors) (4.5 GB)

- [model-00018-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00018-of-00059.safetensors) (4.5 GB)

- [model-00019-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00019-of-00059.safetensors) (4.5 GB)

- [model-00020-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00020-of-00059.safetensors) (4.5 GB)

- [model-00021-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00021-of-00059.safetensors) (4.5 GB)

- [model-00022-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00022-of-00059.safetensors) (4.5 GB)

- [model-00023-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00023-of-00059.safetensors) (4.6 GB)

- [model-00024-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00024-of-00059.safetensors) (4.7 GB)

- [model-00025-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00025-of-00059.safetensors) (4.7 GB)

- [model-00026-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00026-of-00059.safetensors) (4.6 GB)

- [model-00027-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00027-of-00059.safetensors) (4.5 GB)

- [model-00028-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00028-of-00059.safetensors) (4.5 GB)

- [model-00029-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00029-of-00059.safetensors) (4.5 GB)

- [model-00030-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00030-of-00059.safetensors) (4.5 GB)

- [model-00031-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00031-of-00059.safetensors) (4.5 GB)

- [model-00032-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00032-of-00059.safetensors) (4.5 GB)

- [model-00033-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00033-of-00059.safetensors) (4.5 GB)

- [model-00034-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00034-of-00059.safetensors) (4.5 GB)

- [model-00035-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00035-of-00059.safetensors) (4.5 GB)

- [model-00036-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00036-of-00059.safetensors) (4.5 GB)

- [model-00037-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00037-of-00059.safetensors) (4.5 GB)

- [model-00038-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00038-of-00059.safetensors) (4.5 GB)

- [model-00039-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00039-of-00059.safetensors) (4.5 GB)

- [model-00040-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00040-of-00059.safetensors) (4.5 GB)

- [model-00041-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00041-of-00059.safetensors) (4.5 GB)

- [model-00042-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00042-of-00059.safetensors) (4.5 GB)

- [model-00043-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00043-of-00059.safetensors) (4.5 GB)

- [model-00044-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00044-of-00059.safetensors) (4.5 GB)

- [model-00045-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00045-of-00059.safetensors) (4.5 GB)

- [model-00046-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00046-of-00059.safetensors) (4.5 GB)

- [model-00047-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00047-of-00059.safetensors) (4.5 GB)

- [model-00048-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00048-of-00059.safetensors) (4.5 GB)

- [model-00049-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00049-of-00059.safetensors) (4.5 GB)

- [model-00050-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00050-of-00059.safetensors) (4.5 GB)

- [model-00051-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00051-of-00059.safetensors) (4.6 GB)

- [model-00052-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00052-of-00059.safetensors) (4.7 GB)

- [model-00053-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00053-of-00059.safetensors) (4.7 GB)

- [model-00054-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00054-of-00059.safetensors) (4.6 GB)

- [model-00055-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00055-of-00059.safetensors) (4.5 GB)

- [model-00056-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00056-of-00059.safetensors) (4.5 GB)

- [model-00057-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00057-of-00059.safetensors) (4.5 GB)

- [model-00058-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00058-of-00059.safetensors) (4.5 GB)

- [model-00059-of-00059.safetensors](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model-00059-of-00059.safetensors) (1.1 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/model.safetensors.index.json) (161.8 KB)

- [sentencepiece.bpe.model](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/sentencepiece.bpe.model) (481.9 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/special_tokens_map.json) (117.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/tokenizer.json) (1.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/mistralai/Mixtral-8x22B-Instruct-v0.1/tokenizer_config.json) (1.4 KB)


[Back to Main](../../)