
# music-spectrogram-diffusion
---


## README([From Huggingface](https://huggingface.co/google/music-spectrogram-diffusion))



# Multi-instrument Music Synthesis with Spectrogram Diffusion

[Spectrogram Diffusion](https://arxiv.org/abs/2206.05408) by Curtis Hawthorne, Ian Simon, Adam Roberts, Neil Zeghidour, Josh Gardner, Ethan Manilow, and Jesse Engel.

## Abstract

An ideal music synthesizer should be both interactive and expressive, generating high-fidelity audio in realtime for arbitrary combinations of instruments and notes. Recent neural synthesizers have exhibited a tradeoff between domain-specific models that offer detailed control of only specific instruments, or raw waveform models that can train on any music but with minimal control and slow generation. In this work, we focus on a middle ground of neural synthesizers that can generate audio from MIDI sequences with arbitrary combinations of instruments in realtime. This enables training on a wide range of transcription datasets with a single model, which in turn offers note-level control of composition and instrumentation across a wide range of instruments. We use a simple two-stage process: MIDI to spectrograms with an encoder-decoder Transformer, then spectrograms to audio with a generative adversarial network (GAN) spectrogram inverter. We compare training the decoder as an autoregressive model and as a Denoising Diffusion Probabilistic Model (DDPM) and find that the DDPM approach is superior both qualitatively and as measured by audio reconstruction and Fréchet distance metrics. Given the interactivity and generality of this approach, we find this to be a promising first step towards interactive and expressive neural synthesis for arbitrary combinations of instruments and notes.

<img src="https://storage.googleapis.com/music-synthesis-with-spectrogram-diffusion/architecture.png" alt="Architecture diagram">

## Model

As depicted above the model takes as input a MIDI file and tokenizes it into a sequence of 5 second intervals. Each tokenized interval then together with positional encodings is passed through the Note Encoder and its representation is concatenated with the previous window's generated spectrogram representation obtained via the Context Encoder. For the initial 5 second window this is set to zero. The resulting context is then used as conditioning to sample the denoised Spectrogram from the MIDI window and we concatenate this spectrogram to the final output as well as use it for the context of the next MIDI window. The process repeats till we have gone over all the MIDI inputs. Finally a MelGAN decoder converts the potentially long spectrogram to audio which is the final result of this pipeline.

## Example usage

```python
from diffusers import SpectrogramDiffusionPipeline, MidiProcessor

pipe = SpectrogramDiffusionPipeline.from_pretrained("google/music-spectrogram-diffusion")
pipe = pipe.to("cuda")
processor = MidiProcessor()

# Download MIDI from: wget http://www.piano-midi.de/midis/beethoven/beethoven_hammerklavier_2.mid
output = pipe(processor("beethoven_hammerklavier_2.mid"))

audio = output.audios[0]
```



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/README.md) (3.1 KB)

- [continuous_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/continuous_encoder/config.json) (380.0 B)

- [continuous_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/continuous_encoder/model_state.pdparams) (325.2 MB)

- [decoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/decoder/config.json) (329.0 B)

- [decoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/decoder/model_state.pdparams) (910.6 MB)

- [melgan/model.onnx](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/melgan/model.onnx) (57.7 MB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/model_index.json) (449.0 B)

- [notes_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/notes_encoder/config.json) (366.0 B)

- [notes_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/notes_encoder/model_state.pdparams) (334.6 MB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/google/music-spectrogram-diffusion/scheduler/scheduler_config.json) (421.0 B)


[Back to Main](../../)