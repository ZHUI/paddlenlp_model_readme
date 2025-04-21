
# QVQ-72B-Preview
---


## README([From Huggingface](https://huggingface.co/Qwen/QVQ-72B-Preview))




# QVQ-72B-Preview

## Introduction

**QVQ-72B-Preview** is an experimental research model developed by the Qwen team, focusing on enhancing visual reasoning capabilities.

## Performance

|                | **QVQ-72B-Preview** | o1-2024-12-17 | gpt-4o-2024-05-13 | Claude3.5 Sonnet-20241022 | Qwen2VL-72B |
|----------------|-----------------|---------------|-------------------|----------------------------|-------------|
| MMMU(val)      | 70.3            | 77.3          | 69.1              | 70.4                       | 64.5        |
| MathVista(mini) | 71.4            | 71.0          | 63.8              | 65.3                       | 70.5        |
| MathVision(full)   | 35.9            | –             | 30.4              | 35.6                       | 25.9        |
| OlympiadBench  | 20.4            | –             | 25.9              | –                          | 11.2        |


**QVQ-72B-Preview** has achieved remarkable performance on various benchmarks. It scored a remarkable 70.3% on the Multimodal Massive Multi-task Understanding (MMMU) benchmark, showcasing QVQ's powerful ability in multidisciplinary understanding and reasoning. Furthermore, the significant improvements on MathVision highlight the model's progress in mathematical reasoning tasks. OlympiadBench also demonstrates the model's enhanced ability to tackle challenging problems.

***But It's Not All Perfect:  Acknowledging the Limitations***

While **QVQ-72B-Preview** exhibits promising performance that surpasses expectations, it’s important to acknowledge several limitations:

1. **Language Mixing and Code-Switching:** The model might occasionally mix different languages or unexpectedly switch between them, potentially affecting the clarity of its responses.
2. **Recursive Reasoning Loops:**  There's a risk of the model getting caught in recursive reasoning loops, leading to lengthy responses that may not even arrive at a final answer.
3. **Safety and Ethical Considerations:** Robust safety measures are needed to ensure reliable and safe performance. Users should exercise caution when deploying this model.
4. **Performance and Benchmark Limitations:** Despite the improvements in visual reasoning, QVQ doesn’t entirely replace the capabilities of Qwen2-VL-72B. During multi-step visual reasoning, the model might gradually lose focus on the image content, leading to hallucinations. Moreover, QVQ doesn’t show significant improvement over Qwen2-VL-72B in basic recognition tasks like identifying people, animals, or plants.

Note: Currently, the model only supports single-round dialogues and image outputs. It does not support video inputs.
## Quickstart

We offer a toolkit to help you handle various types of visual input more conveniently. This includes base64, URLs, and interleaved images and videos. You can install it using the following command:

```bash
pip install qwen-vl-utils
```

Here we show a code snippet to show you how to use the chat model with `transformers` and `qwen_vl_utils`:

```python
from paddlenlp.transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/QVQ-72B-Preview", torch_dtype="auto", device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/QVQ/demo.png",
            },
            {"type": "text", "text": "What value should be filled in the blank space?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=8192)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{qvq-72b-preview,
    title = {QVQ: To See the World with Wisdom},
    url = {https://qwenlm.github.io/blog/qvq-72b-preview/},
    author = {Qwen Team},
    month = {December},
    year = {2024}
}

@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```



## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/LICENSE) (6.8 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/README.md) (5.9 KB)

- [chat_template.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/chat_template.json) (1.1 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/config.json) (1.1 KB)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/configuration.json) (71.0 B)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/generation_config.json) (210.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/merges.txt) (1.6 MB)

- [model-00001-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00001-of-00038.safetensors) (3.6 GB)

- [model-00002-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00002-of-00038.safetensors) (3.6 GB)

- [model-00003-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00003-of-00038.safetensors) (3.7 GB)

- [model-00004-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00004-of-00038.safetensors) (3.7 GB)

- [model-00005-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00005-of-00038.safetensors) (3.7 GB)

- [model-00006-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00006-of-00038.safetensors) (3.6 GB)

- [model-00007-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00007-of-00038.safetensors) (3.7 GB)

- [model-00008-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00008-of-00038.safetensors) (3.7 GB)

- [model-00009-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00009-of-00038.safetensors) (3.7 GB)

- [model-00010-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00010-of-00038.safetensors) (3.6 GB)

- [model-00011-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00011-of-00038.safetensors) (3.7 GB)

- [model-00012-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00012-of-00038.safetensors) (3.7 GB)

- [model-00013-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00013-of-00038.safetensors) (3.7 GB)

- [model-00014-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00014-of-00038.safetensors) (3.6 GB)

- [model-00015-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00015-of-00038.safetensors) (3.7 GB)

- [model-00016-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00016-of-00038.safetensors) (3.7 GB)

- [model-00017-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00017-of-00038.safetensors) (3.7 GB)

- [model-00018-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00018-of-00038.safetensors) (3.6 GB)

- [model-00019-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00019-of-00038.safetensors) (3.7 GB)

- [model-00020-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00020-of-00038.safetensors) (3.7 GB)

- [model-00021-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00021-of-00038.safetensors) (3.7 GB)

- [model-00022-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00022-of-00038.safetensors) (3.6 GB)

- [model-00023-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00023-of-00038.safetensors) (3.7 GB)

- [model-00024-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00024-of-00038.safetensors) (3.7 GB)

- [model-00025-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00025-of-00038.safetensors) (3.7 GB)

- [model-00026-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00026-of-00038.safetensors) (3.6 GB)

- [model-00027-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00027-of-00038.safetensors) (3.7 GB)

- [model-00028-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00028-of-00038.safetensors) (3.7 GB)

- [model-00029-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00029-of-00038.safetensors) (3.7 GB)

- [model-00030-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00030-of-00038.safetensors) (3.6 GB)

- [model-00031-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00031-of-00038.safetensors) (3.7 GB)

- [model-00032-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00032-of-00038.safetensors) (3.7 GB)

- [model-00033-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00033-of-00038.safetensors) (3.7 GB)

- [model-00034-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00034-of-00038.safetensors) (3.6 GB)

- [model-00035-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00035-of-00038.safetensors) (3.7 GB)

- [model-00036-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00036-of-00038.safetensors) (3.7 GB)

- [model-00037-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00037-of-00038.safetensors) (2.1 GB)

- [model-00038-of-00038.safetensors](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model-00038-of-00038.safetensors) (2.3 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/model.safetensors.index.json) (105.2 KB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/preprocessor_config.json) (348.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/tokenizer.json) (6.7 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/tokenizer_config.json) (5.6 KB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Qwen/QVQ-72B-Preview/vocab.json) (2.6 MB)


[Back to Main](../../)