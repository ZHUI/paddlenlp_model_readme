
# text-to-video-ms-1.7b
---


## README([From Huggingface](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b))



# Text-to-video-synthesis Model in Open Domain

This model is based on a multi-stage text-to-video generation diffusion model, which inputs a description text and returns a video that matches the text description. Only English input is supported.

**We Are Hiring!** (Based in Beijing / Hangzhou, China.)

If you're looking for an exciting challenge and the opportunity to work with cutting-edge technologies in AIGC and large-scale pretraining, then we are the place for you. We are looking for talented, motivated and creative individuals to join our team. If you are interested, please send your CV to us.

EMAIL: yingya.zyy@alibaba-inc.com

## Model description

The text-to-video generation diffusion model consists of three sub-networks: text feature extraction model, text feature-to-video latent space diffusion model, and video latent space to video visual space model. The overall model parameters are about 1.7 billion. Currently, it only supports English input. The diffusion model adopts a UNet3D structure, and implements video generation through the iterative denoising process from the pure Gaussian noise video.

This model is meant for research purposes. Please look at the [model limitations and biases and misuse](#model-limitations-and-biases), [malicious use and excessive use](#misuse-malicious-use-and-excessive-use) sections.

## Model Details

- **Developed by:** [ModelScope](https://modelscope.cn/)
- **Model type:** Diffusion-based text-to-video generation model
- **Language(s):** English
- **License:**[ CC-BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- **Resources for more information:** [ModelScope GitHub Repository](https://github.com/modelscope/modelscope), [Summary](https://modelscope.cn/models/damo/text-to-video-synthesis/summary).
- **Cite as:**

## Use cases

This model has a wide range of applications and can reason and generate videos based on arbitrary English text descriptions. 

## Usage 

Let's first install the libraries required:

```bash
$ pip install diffusers transformers accelerate torch
```

Now, generate a video:

```python
import paddle
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", dtype=paddle.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = export_to_video(video_frames)
```

Here are some results:

<table>
    <tr>
        <td><center>
        An astronaut riding a horse.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astr.gif"
            alt="An astronaut riding a horse."
            style="width: 300px;" />
        </center></td>
        <td ><center>
        Darth vader surfing in waves.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vader.gif"
            alt="Darth vader surfing in waves."
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

## Long Video Generation

You can optimize for memory usage by enabling attention and VAE slicing and using Torch 2.0.
This should allow you to generate videos up to 25 seconds on less than 16GB of GPU VRAM.

```bash
$ pip install git+https://github.com/huggingface/diffusers transformers accelerate
```

```py
import paddle
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# load pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", dtype=paddle.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# generate
prompt = "Spiderman is surfing. Darth Vader is also surfing and following Spiderman"
video_frames = pipe(prompt, num_inference_steps=25, num_frames=200).frames

# convent to video
video_path = export_to_video(video_frames)
```


## View results

The above code will display the save path of the output video, and the current encoding format can be played with [VLC player](https://www.videolan.org/vlc/).

The output mp4 file can be viewed by [VLC media player](https://www.videolan.org/vlc/). Some other media players may not view it normally.

## Model limitations and biases

* The model is trained based on public data sets such as Webvid, and the generated results may have deviations related to the distribution of training data.
* This model cannot achieve perfect film and television quality generation.
* The model cannot generate clear text.
* The model is mainly trained with English corpus and does not support other languages ​​at the moment**.
* The performance of this model needs to be improved on complex compositional generation tasks.

## Misuse, Malicious Use and Excessive Use

* The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities.
* It is prohibited to generate content that is demeaning or harmful to people or their environment, culture, religion, etc.
* Prohibited for pornographic, violent and bloody content generation.
* Prohibited for error and false information generation.

## Training data

The training data includes [LAION5B](https://huggingface.co/datasets/laion/laion2B-en), [ImageNet](https://www.image-net.org/), [Webvid](https://m-bain.github.io/webvid-dataset/) and other public datasets. Image and video filtering is performed after pre-training such as aesthetic score, watermark score, and deduplication.

_(Part of this model card has been taken from [here](https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis))_

## Citation

```bibtex
    @article{wang2023modelscope,
      title={Modelscope text-to-video technical report},
      author={Wang, Jiuniu and Yuan, Hangjie and Chen, Dayou and Zhang, Yingya and Wang, Xiang and Zhang, Shiwei},
      journal={arXiv preprint arXiv:2308.06571},
      year={2023}
    }
    @InProceedings{VideoFusion,
        author    = {Luo, Zhengxiong and Chen, Dayou and Zhang, Yingya and Huang, Yan and Wang, Liang and Shen, Yujun and Zhao, Deli and Zhou, Jingren and Tan, Tieniu},
        title     = {VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023}
    }
```




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/README.md) (6.7 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/model_index.json) (441.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/scheduler/scheduler_config.json) (496.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/text_encoder/config.json) (718.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/text_encoder/model_state.pdparams) (1.3 GB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/tokenizer/special_tokens_map.json) (377.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/tokenizer/tokenizer_config.json) (793.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/unet/config.json) (853.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/unet/model_state.pdparams) (5.3 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/vae/config.json) (829.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/damo-vilab/text-to-video-ms-1.7b/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)