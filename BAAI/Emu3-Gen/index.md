
# Emu3-Gen
---


## README([From Huggingface](https://huggingface.co/BAAI/Emu3-Gen))

---
pipeline_tag: any-to-any
license: apache-2.0
library_name: transformers
---

<div align='center'>
<h1>Emu3: Next-Token Prediction is All You Need</h1h1>
<h3></h3>

[Emu3 Team, BAAI](https://www.baai.ac.cn/english.html)

| [Project Page](https://emu.baai.ac.cn) | [Paper](https://huggingface.co/papers/2409.18869) | [🤗HF Models](https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f) |  [github](https://github.com/baaivision/Emu3)
 | [Demo](https://huggingface.co/spaces/BAAI/Emu3) |


</div>

<div align='center'>
<img src="https://github.com/baaivision/Emu3/blob/main/assets/arch.png?raw=True" class="interpolation-image" alt="arch." height="80%" width="70%" />
</div>

We introduce **Emu3**, a new suite of state-of-the-art multimodal models trained solely with **<i>next-token prediction</i>**! By tokenizing images, text, and videos into a discrete space, we train a single transformer from scratch on a mixture of multimodal sequences.

### Emu3 excels in both generation and perception
**Emu3** outperforms several well-established task-specific models in both generation and perception tasks, surpassing flagship open models such as SDXL, LLaVA-1.6 and OpenSora-1.2, while eliminating the need for diffusion or compositional architectures.

<div align='center'>
<img src="https://github.com/baaivision/Emu3/blob/main/assets/comparison.png?raw=True" class="interpolation-image" alt="comparison." height="80%" width="80%" />
</div>

### Highlights

- **Emu3** is capable of generating high-quality images following the text input, by simply predicting the next vision token. The model naturally supports flexible resolutions and styles.
- **Emu3** shows strong vision-language understanding capabilities to see the physical world and provides coherent text responses. Notably, this capability is achieved without depending on a CLIP and a pretrained LLM.
- **Emu3** simply generates a video causally by predicting the next token in a video sequence, unlike the video diffusion model as in Sora. With a video in context, Emu3 can also naturally extend the video and predict what will happen next. 



#### Quickstart

```python
from PIL import Image
from paddlenlp.transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from paddlenlp.transformers.generation.configuration_utils import GenerationConfig
from paddlenlp.transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch

import sys
sys.path.append(PATH_TO_BAAI_Emu3-Gen_MODEL)
from processing_emu3 import Emu3Processor

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

# prepare model and processor
model = AutoModelForCausalLM.from_pretrained(
    EMU_HUB,
    device_map="cuda:0",
    dtype=paddle.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
prompt = "a portrait of young girl."
prompt += POSITIVE_PROMPT

kwargs = dict(
    mode='G',
    ratio="1:1",
    image_area=model.config.image_area,
    return_tensors="pd",
    padding="longest",
)
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=40960,
    do_sample=True,
    top_k=2048,
)

h = pos_inputs.image_size[:, 0]
w = pos_inputs.image_size[:, 1]
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList([
    UnbatchedClassifierFreeGuidanceLogitsProcessor(
        classifier_free_guidance,
        model,
        unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
    ),
    PrefixConstrainedLogitsProcessor(
        constrained_fn ,
        num_beams=1,
    ),
])

# generate
outputs = model.generate(
    pos_inputs.input_ids.to("cuda:0"),
    GENERATION_CONFIG,
    logits_processor=logits_processor,
    attention_mask=pos_inputs.attention_mask.to("cuda:0"),
)[0]

mm_list = processor.decode(outputs[0])
for idx, im in enumerate(mm_list):
    if not isinstance(im, Image.Image):
        continue
    im.save(f"result_{idx}.png")
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/README.md) (4.8 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/config.json) (891.0 B)

- [emu3.tiktoken](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/emu3.tiktoken) (2.4 MB)

- [emu3_vision_tokens.txt](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/emu3_vision_tokens.txt) (768.0 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/generation_config.json) (147.0 B)

- [model-00001-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/model-00001-of-00007.safetensors) (4.6 GB)

- [model-00002-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/model-00002-of-00007.safetensors) (4.5 GB)

- [model-00003-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/model-00003-of-00007.safetensors) (4.7 GB)

- [model-00004-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/model-00004-of-00007.safetensors) (4.7 GB)

- [model-00005-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/model-00005-of-00007.safetensors) (4.5 GB)

- [model-00006-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/model-00006-of-00007.safetensors) (4.7 GB)

- [model-00007-of-00007.safetensors](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/model-00007-of-00007.safetensors) (4.1 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/model.safetensors.index.json) (23.4 KB)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/special_tokens_map.json) (99.0 B)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/BAAI/Emu3-Gen/tokenizer_config.json) (364.0 B)


[Back to Main](../../)