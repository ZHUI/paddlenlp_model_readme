
# audioldm
---


## README([From Huggingface](https://huggingface.co/cvssp/audioldm))



# AudioLDM

AudioLDM is a latent text-to-audio diffusion model capable of generating realistic audio samples given any text input. It is available in the 🧨 Diffusers library from v0.15.0 onwards.

# Model Details

AudioLDM was proposed in the paper [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://arxiv.org/abs/2301.12503) by Haohe Liu et al.

Inspired by [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4), AudioLDM
is a text-to-audio _latent diffusion model (LDM)_ that learns continuous audio representations from [CLAP](https://huggingface.co/laion/clap-htsat-unfused)
latents. AudioLDM takes a text prompt as input and predicts the corresponding audio. It can generate text-conditional
sound effects, human speech and music.

# Checkpoint Details

This is the original, **small** version of the AudioLDM model, also referred to as **audioldm-s-full**. The four AudioLDM checkpoints are summarised in the table below:

**Table 1:** Summary of the AudioLDM checkpoints.

| Checkpoint                                                            | Training Steps | Audio conditioning | CLAP audio dim | UNet dim | Params |
|-----------------------------------------------------------------------|----------------|--------------------|----------------|----------|--------|
| [audioldm-s-full](https://huggingface.co/cvssp/audioldm)              | 1.5M           | No                 | 768            | 128      | 421M   |
| [audioldm-s-full-v2](https://huggingface.co/cvssp/audioldm-s-full-v2) | > 1.5M         | No                 | 768            | 128      | 421M   |
| [audioldm-m-full](https://huggingface.co/cvssp/audioldm-m-full)       | 1.5M           | Yes                | 1024           | 192      | 652M   |
| [audioldm-l-full](https://huggingface.co/cvssp/audioldm-l-full)       | 1.5M           | No                 | 768            | 256      | 975M   |

## Model Sources

- [**Original Repository**](https://github.com/haoheliu/AudioLDM)
- [**🧨 Diffusers Pipeline**](https://huggingface.co/docs/diffusers/api/pipelines/audioldm)
- [**Paper**](https://arxiv.org/abs/2301.12503)
- [**Demo**](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)

# Usage

First, install the required packages:

```
pip install --upgrade diffusers transformers accelerate
```

## Text-to-Audio

For text-to-audio generation, the [AudioLDMPipeline](https://huggingface.co/docs/diffusers/api/pipelines/audioldm) can be 
used to load pre-trained weights and generate text-conditional audio outputs:

```python
from diffusers import AudioLDMPipeline
import torch

repo_id = "cvssp/audioldm"
pipe = AudioLDMPipeline.from_pretrained(repo_id, dtype=paddle.float16)
pipe = pipe.to("cuda")

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
```

The resulting audio output can be saved as a .wav file:
```python
import scipy

scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
```

Or displayed in a Jupyter Notebook / Google Colab:
```python
from IPython.display import Audio

Audio(audio, rate=16000)
```
<audio controls>
  <source src="https://huggingface.co/datasets/sanchit-gandhi/audioldm-readme-samples/resolve/main/audioldm-techno.wav" type="audio/wav">
Your browser does not support the audio element.
</audio>

## Tips

Prompts:
* Descriptive prompt inputs work best: you can use adjectives to describe the sound (e.g. "high quality" or "clear") and make the prompt context specific (e.g., "water stream in a forest" instead of "stream").
* It's best to use general terms like 'cat' or 'dog' instead of specific names or abstract objects that the model may not be familiar with.

Inference:
* The _quality_ of the predicted audio sample can be controlled by the `num_inference_steps` argument: higher steps give higher quality audio at the expense of slower inference.
* The _length_ of the predicted audio sample can be controlled by varying the `audio_length_in_s` argument.

# Citation

**BibTeX:**
```
@article{liu2023audioldm,
  title={AudioLDM: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2301.12503},
  year={2023}
}
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/README.md) (4.3 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/model_index.json) (494.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/scheduler/scheduler_config.json) (436.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/text_encoder/config.json) (971.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/text_encoder/model_state.pdparams) (478.0 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/tokenizer/merges.txt) (445.7 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/tokenizer/special_tokens_map.json) (239.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/tokenizer/tokenizer_config.json) (434.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/tokenizer/vocab.json) (779.6 KB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/unet/config.json) (1.6 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/unet/model_state.pdparams) (705.9 MB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/vae/config.json) (634.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/vae/model_state.pdparams) (211.3 MB)

- [vocoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/vocoder/config.json) (807.0 B)

- [vocoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/cvssp/audioldm/vocoder/model_state.pdparams) (210.8 MB)


[Back to Main](../../)