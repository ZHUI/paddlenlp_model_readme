
# Paint-by-Example
---


## README([From Huggingface](https://huggingface.co/Fantasy-Studio/Paint-by-Example))


# Paint-By-Example

## Overview

[Paint by Example: Exemplar-based Image Editing with Diffusion Models](https://arxiv.org/abs/2211.13227) by Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, Fang Wen

The abstract of the paper is the following:

*Language-guided image editing has achieved great success recently. In this paper, for the first time, we investigate exemplar-guided image editing for more precise control. We achieve this goal by leveraging self-supervised training to disentangle and re-organize the source image and the exemplar. However, the naive approach will cause obvious fusing artifacts. We carefully analyze it and propose an information bottleneck and strong augmentations to avoid the trivial solution of directly copying and pasting the exemplar image. Meanwhile, to ensure the controllability of the editing process, we design an arbitrary shape mask for the exemplar image and leverage the classifier-free guidance to increase the similarity to the exemplar image. The whole framework involves a single forward of the diffusion model without any iterative optimization. We demonstrate that our method achieves an impressive performance and enables controllable editing on in-the-wild images with high fidelity.*

The original codebase can be found [here](https://github.com/Fantasy-Studio/Paint-by-Example).

## Available Pipelines:

| Pipeline | Tasks | Colab
|---|---|:---:|
| [pipeline_paint_by_example.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/paint_by_example/pipeline_paint_by_example.py) | *Image-Guided Image Painting* | - |

## Tips

- [Fantasy-Studio/Paint-by-Example](https://huggingface.co/Fantasy-Studio/Paint-by-Example) has been warm-started from the [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) and with the objective to inpaint partly masked images conditioned on example / reference images
- To quickly demo *PaintByExample*, please have a look at [this demo](https://huggingface.co/spaces/Fantasy-Studio/Paint-by-Example).
- You can run the following code snippet as an example:

```python
# !pip install diffusers transformers

import PIL
import requests
import paddle
from io import BytesIO
from diffusers import DiffusionPipeline


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
mask_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_1.png"
example_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
example_image = download_image(example_url).resize((512, 512))

pipe = DiffusionPipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    dtype=paddle.float16,
)
pipe = pipe

image = pipe(image=init_image, mask_image=mask_image, example_image=example_image).images[0]
image
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/README.md) (3.2 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/feature_extractor/preprocessor_config.json) (518.0 B)

- [image_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/image_encoder/config.json) (524.0 B)

- [image_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/image_encoder/model_config.json) (262.0 B)

- [image_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/image_encoder/model_state.pdparams) (1.4 GB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/model_index.json) (510.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/scheduler/scheduler_config.json) (343.0 B)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/unet/config.json) (1.1 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/vae/config.json) (703.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/Paint-by-Example/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)