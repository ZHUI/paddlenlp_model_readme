
# t2iadapter_color_sd14v1
---


## README([From Huggingface](https://huggingface.co/TencentARC/t2iadapter_color_sd14v1))



# T2I Adapter - Color

T2I Adapter is a network providing additional conditioning to stable diffusion. Each t2i checkpoint takes a different type of conditioning as input and is used with a specific base stable diffusion checkpoint.

This checkpoint provides conditioning on color palettes for the stable diffusion 1.4 checkpoint.

## Model Details
- **Developed by:** T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models
- **Model type:** Diffusion-based text-to-image generation model
- **Language(s):** English
- **License:** Apache 2.0
- **Resources for more information:** [GitHub Repository](https://github.com/TencentARC/T2I-Adapter), [Paper](https://arxiv.org/abs/2302.08453).
- **Cite as:**

  @misc{
    title={T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models}, 
    author={Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, Xiaohu Qie},
    year={2023},
    eprint={2302.08453},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }

### Checkpoints

| Model Name | Control Image Overview| Control Image Example | Generated Image Example |
|---|---|---|---|
|[TencentARC/t2iadapter_color_sd14v1](https://huggingface.co/TencentARC/t2iadapter_color_sd14v1)<br/> *Trained with spatial color palette* | A image with 8x8 color palette.|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_sample_input.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_sample_input.png"/></a>|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_sample_output.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_sample_output.png"/></a>|
|[TencentARC/t2iadapter_canny_sd14v1](https://huggingface.co/TencentARC/t2iadapter_canny_sd14v1)<br/> *Trained with canny edge detection* | A monochrome image with white edges on a black background.|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/canny_sample_input.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/canny_sample_input.png"/></a>|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/canny_sample_output.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/canny_sample_output.png"/></a>|
|[TencentARC/t2iadapter_sketch_sd14v1](https://huggingface.co/TencentARC/t2iadapter_sketch_sd14v1)<br/> *Trained with [PidiNet](https://github.com/zhuoinoulu/pidinet) edge detection* | A hand-drawn monochrome image with white outlines on a black background.|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/sketch_sample_input.png"><img width="64" style="margin:0;padding:0;" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/sketch_sample_input.png"/></a>|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/sketch_sample_output.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/sketch_sample_output.png"/></a>|
|[TencentARC/t2iadapter_depth_sd14v1](https://huggingface.co/TencentARC/t2iadapter_depth_sd14v1)<br/> *Trained with Midas depth estimation*  | A grayscale image with black representing deep areas and white representing shallow areas.|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_input.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_input.png"/></a>|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_output.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_output.png"/></a>|
|[TencentARC/t2iadapter_openpose_sd14v1](https://huggingface.co/TencentARC/t2iadapter_openpose_sd14v1)<br/> *Trained with OpenPose bone image*  | A [OpenPose bone](https://github.com/CMU-Perceptual-Computing-Lab/openpose) image.|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/openpose_sample_input.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/openpose_sample_input.png"/></a>|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/openpose_sample_output.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/openpose_sample_output.png"/></a>|
|[TencentARC/t2iadapter_keypose_sd14v1](https://huggingface.co/TencentARC/t2iadapter_keypose_sd14v1)<br/> *Trained with mmpose skeleton image*  | A [mmpose skeleton](https://github.com/open-mmlab/mmpose) image.|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_input.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_input.png"/></a>|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_output.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_output.png"/></a>|
|[TencentARC/t2iadapter_seg_sd14v1](https://huggingface.co/TencentARC/t2iadapter_seg_sd14v1)<br/>*Trained with semantic segmentation*  | An [custom](https://github.com/TencentARC/T2I-Adapter/discussions/25) segmentation protocol image.|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/seg_sample_input.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/seg_sample_input.png"/></a>|<a href="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/seg_sample_output.png"><img width="64" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/seg_sample_output.png"/></a> |
|[TencentARC/t2iadapter_canny_sd15v2](https://huggingface.co/TencentARC/t2iadapter_canny_sd15v2)||
|[TencentARC/t2iadapter_depth_sd15v2](https://huggingface.co/TencentARC/t2iadapter_depth_sd15v2)||
|[TencentARC/t2iadapter_sketch_sd15v2](https://huggingface.co/TencentARC/t2iadapter_sketch_sd15v2)||
|[TencentARC/t2iadapter_zoedepth_sd15v1](https://huggingface.co/TencentARC/t2iadapter_zoedepth_sd15v1)||

## Example

1. Dependencies

```sh
pip install diffusers transformers
```

2. Run code:

```python
from PIL import Image
import paddle
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

image = Image.open('./images/color_ref.png')

color_palette = image.resize((8, 8))
color_palette = color_palette.resize((512, 512), resample=Image.Resampling.NEAREST)

color_palette.save('./images/color_palette.png')

adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_color_sd14v1", dtype=paddle.float16)
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    adapter=adapter,
    dtype=paddle.float16,
)
pipe

generator = paddle.seed(0)

out_image = pipe(
    "At night, glowing cubes in front of the beach",
    image=color_palette,
    generator=generator,
).images[0]

out_image.save('./images/color_out_image.png')
```


![![color_ref](https://huggingface.co/TencentARC/t2iadapter_color_sd14v1/resolve/main/./images/color_ref.png)
![![color_palette](https://huggingface.co/TencentARC/t2iadapter_color_sd14v1/resolve/main/./images/color_palette.png)
![![color_out_image](https://huggingface.co/TencentARC/t2iadapter_color_sd14v1/resolve/main/./images/color_out_image.png)



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/TencentARC/t2iadapter_color_sd14v1/README.md) (7.7 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/TencentARC/t2iadapter_color_sd14v1/config.json) (278.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/TencentARC/t2iadapter_color_sd14v1/model_state.pdparams) (71.3 MB)


[Back to Main](../../)