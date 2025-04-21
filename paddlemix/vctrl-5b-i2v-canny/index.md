
# vctrl-5b-i2v-canny
---


## README([From Huggingface](https://huggingface.co/paddlemix/vctrl-5b-i2v-canny))



English | [简体中文](README_zh.md)
# VCtrl
<p style="text-align: center;">
  <p align="center"> 
  <a href="https://huggingface.co/PaddleMIX">🤗 Huggingface Space</a> |
  <a href="https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/ppvctrl">🌐 Github </a> | 
  <a href="">📜 arxiv </a> |
  <a href="https://pp-vctrl.github.io/">📷 Project </a> 
</p>

## Model Introduction
**VCtrl** is a versatile video generation control model that introduces an auxiliary conditional encoder to flexibly connect with various control modules while avoiding large-scale retraining of the original generator. The model efficiently transmits control signals through sparse residual connections and standardizes diverse control inputs into a unified representation via a consistent encoding process. Task-specific masks are further incorporated to enhance adaptability. Thanks to this unified and flexible design, VCtrl can be widely applied in ​**character animation**, ​**scene transition**, ​**video editing**, and other video generation scenarios. The table below provides detailed information about the video generation models we offer:

<table  style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">Model Name</th>
    <th style="text-align: center;">VCtrl-Canny</th>
    <th style="text-align: center;">VCtrl-Mask</th>
    <th style="text-align: center;">VCtrl-Pose</th>
  </tr>
  <tr>
    <td style="text-align: center;">Video Resolution</td>
    <td colspan="1" style="text-align: center;">720 * 480</td>
    <td colspan="1" style="text-align: center;"> 720 * 480 </td>
    <td colspan="1 style="text-align: center;"> 720 * 480 & 480 * 720 </td>
    </tr>
  <tr>
    <td style="text-align: center;">Inference Precision</td>
    <td colspan="3" style="text-align: center;"><b>FP16(Recommended)</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">Single GPU VRAM Usage</td>
    <td colspan="3"  style="text-align: center;"><b>V100: 32GB minimum*</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">Inference Speed<br>(Step = 25, FP16)</td>
    <td colspan="3" style="text-align: center;">Single A100: ~300s(49 frames)<br>Single V100: ~400s(49 frames)</td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Language</td>
    <td colspan="5" style="text-align: center;">English*</td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Length Limit</td>
    <td colspan="3" style="text-align: center;">224 Tokens</td>
  </tr>
  <tr>
    <td style="text-align: center;">Video Length</td>
    <td colspan="3" style="text-align: center;">T2V model supports only 49 frames, I2V model can extend to any frame count</td>
  </tr>
  <tr>
    <td style="text-align: center;">Frame Rate</td>
    <td colspan="3" style="text-align: center;">30 FPS </td>
  </tr>
</table>

## Quick Start 🤗

This model is now supported for deployment using the ppdiffusers library from paddlemix. Follow the steps below to get started.

**We recommend visiting our [github](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/ppvctrl) for a better experience.**

1. Install dependencies

```shell
# Clone the PaddleMIX repository
git clone https://github.com/PaddlePaddle/PaddleMIX.git
# Install paddlemix
cd PaddleMIX
pip install -e .
# Install ppdiffusers
pip install -e ppdiffusers
# Install paddlenlp
pip install paddlenlp==v3.0.0-beta2
# Navigate to the vctrl directory
cd ppdiffusers/examples/ppvctrl
# Install other required dependencies
pip install -r requirements.txt
# Install paddlex
pip install paddlex==3.0.0b2
```

2. Run the code

```python
import os
import paddle
import numpy as np
from decord import VideoReader
from moviepy.editor import ImageSequenceClip
from PIL import Image
from ppdiffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXTransformer3DVCtrlModel,
    CogVideoXVCtrlPipeline,
    VCtrlModel,
)
def write_mp4(video_path, samples, fps=8):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac")


def save_vid_side_by_side(batch_output, validation_control_images, output_folder, fps):
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    ori_video_path = output_folder + "/origin_predict.mp4"
    video_path = output_folder + "/test_1.mp4"
    ori_final_images = []
    final_images = []
    outputs = []

    def get_concat_h(im1, im2):
        dst = Image.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    for image_list in zip(validation_control_images, flattened_batch_output):
        predict_img = image_list[1].resize(image_list[0].size)
        result = get_concat_h(image_list[0], predict_img)
        ori_final_images.append(np.array(image_list[1]))
        final_images.append(np.array(result))
        outputs.append(np.array(predict_img))
    write_mp4(ori_video_path, ori_final_images, fps=fps)
    write_mp4(video_path, final_images, fps=fps)
    output_path = output_folder + "/output.mp4"
    write_mp4(output_path, outputs, fps=fps)


def load_images_from_folder_to_pil(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    def frame_number(filename):
        new_pattern_match = re.search("frame_(\\d+)_7fps", filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))
        matches = re.findall("\\d+", filename)
        if matches:
            if matches[-1] == "0000" and len(matches) > 1:
                return int(matches[-2])
            return int(matches[-1])
        return float("inf")

    sorted_files = sorted(os.listdir(folder), key=frame_number)
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)
    return images


def load_images_from_video_to_pil(video_path):
    images = []
    vr = VideoReader(video_path)
    length = len(vr)
    for idx in range(length):
        frame = vr[idx].asnumpy()
        images.append(Image.fromarray(frame))
    return images


validation_control_images = load_images_from_video_to_pil('your_path')
prompt = 'Group of fishes swimming in aquarium.'
vctrl = VCtrlModel.from_pretrained(
            paddlemix/vctrl-5b-t2v-canny,
            low_cpu_mem_usage=True,
            paddle_dtype=paddle.float16
        )
pipeline = CogVideoXVCtrlPipeline.from_pretrained(
            paddlemix/cogvideox-5b-vctrl, 
            vctrl=vctrl, 
            paddle_dtype=paddle.float16, 
            low_cpu_mem_usage=True,
            map_location="cpu",
        )
pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
pipeline.vae.enable_tiling()
pipeline.vae.enable_slicing()
task='canny'
final_result=[]
video = pipeline(
        prompt=prompt,
        num_inference_steps=25,
        num_frames=49,
        guidance_scale=35,
        generator=paddle.Generator().manual_seed(42),
        conditioning_frames=validation_control_images[:num_frames],
        conditioning_frame_indices=list(range(num_frames)),
        conditioning_scale=1.0,
        width=720,
        height=480,
        task='canny',
        conditioning_masks=validation_mask_images[:num_frames] if task == "mask" else None,
        vctrl_layout_type='spacing',
    ).frames[0]
final_result.append(video)
save_vid_side_by_side(final_result, validation_control_images[:num_frames], 'save.mp4', fps=30)
```

## In-Depth Exploration

Welcome to our [github]("https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/ppvctrl"), where you will find：

1. More detailed technical explanations and code walkthroughs.
2. Algorithm details for extracting control conditions.
3. Detailed code for model inference.
4. Project update logs and more interactive opportunities.
5. PaddleMix toolchain to help you better utilize the model.

<!-- ## Citation

```
@article{yang2024cogvideox,
  title={VCtrl: Enabling Versatile Controls for Video Diffusion Models},
  year={2025}
}
``` -->



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/vctrl-5b-i2v-canny/README.md) (8.2 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/vctrl-5b-i2v-canny/config.json) (502.0 B)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/vctrl-5b-i2v-canny/model_state.pdparams) (1.4 GB)


[Back to Main](../../)