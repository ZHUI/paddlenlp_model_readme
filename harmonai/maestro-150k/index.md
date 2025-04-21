
# maestro-150k
---


## README([From Huggingface](https://huggingface.co/harmonai/maestro-150k))



[Dance Diffusion](https://github.com/Harmonai-org/sample-generator) is now available in 🧨 Diffusers.

## FP32

```python
# !pip install diffusers[torch] accelerate scipy
from diffusers import DiffusionPipeline
from scipy.io.wavfile import write

model_id = "harmonai/maestro-150k"
pipe = DiffusionPipeline.from_pretrained(model_id)
pipe = pipe

audios = pipe(audio_length_in_s=4.0).audios

# To save locally
for i, audio in enumerate(audios):
    write(f"maestro_test_{i}.wav", pipe.unet.sample_rate, audio.transpose())
    
# To dislay in google colab
import IPython.display as ipd
for audio in audios:
    display(ipd.Audio(audio, rate=pipe.unet.sample_rate))
```

## FP16

Faster at a small loss of quality

```python
# !pip install diffusers[torch] accelerate scipy
from diffusers import DiffusionPipeline
from scipy.io.wavfile import write
import paddle

model_id = "harmonai/maestro-150k"
pipe = DiffusionPipeline.from_pretrained(model_id, dtype=paddle.float16)
pipe = pipe

audios = pipeline(audio_length_in_s=4.0).audios

# To save locally
for i, audio in enumerate(audios):
    write(f"maestro_test_{i}.wav", pipe.unet.sample_rate, audio.transpose())
    
# To dislay in google colab
import IPython.display as ipd
for audio in audios:
    display(ipd.Audio(audio, rate=pipe.unet.sample_rate))
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/harmonai/maestro-150k/README.md) (1.3 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/harmonai/maestro-150k/model_index.json) (202.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/harmonai/maestro-150k/scheduler/scheduler_config.json) (108.0 B)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/harmonai/maestro-150k/unet/config.json) (1.2 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/harmonai/maestro-150k/unet/model_state.pdparams) (844.6 MB)


[Back to Main](../../)