
# Realistic_Vision_V2.0
---


## README([From Huggingface](https://huggingface.co/SG161222/Realistic_Vision_V2.0))


<strong>Check my exclusive models on Mage: </strong><a href="https://www.mage.space/play/4371756b27bf52e7a1146dc6fe2d969c" rel="noopener noreferrer nofollow"><strong>ParagonXL</strong></a><strong> / </strong><a href="https://www.mage.space/play/df67a9f27f19629a98cb0fb619d1949a" rel="noopener noreferrer nofollow"><strong>NovaXL</strong></a><strong> / </strong><a href="https://www.mage.space/play/d8db06ae964310acb4e090eec03984df" rel="noopener noreferrer nofollow"><strong>NovaXL Lightning</strong></a><strong> / </strong><a href="https://www.mage.space/play/541da1e10976ab82976a5cacc770a413" rel="noopener noreferrer nofollow"><strong>NovaXL V2</strong></a><strong> / </strong><a href="https://www.mage.space/play/a56d2680c464ef25b8c66df126b3f706" rel="noopener noreferrer nofollow"><strong>NovaXL Pony</strong></a><strong> / </strong><a href="https://www.mage.space/play/b0ab6733c3be2408c93523d57a605371" rel="noopener noreferrer nofollow"><strong>NovaXL Pony Lightning</strong></a><strong> / </strong><a href="https://www.mage.space/play/e3b01cd493ed86ed8e4708751b1c9165" rel="noopener noreferrer nofollow"><strong>RealDreamXL</strong></a><strong> / </strong><a href="https://www.mage.space/play/ef062fc389c3f8723002428290c1158c" rel="noopener noreferrer nofollow"><strong>RealDreamXL Lightning</strong></a></p>
<b>This model is available on <a href="https://www.mage.space/">Mage.Space</a> (main sponsor)</b><br>
<b>You can support me directly on Boosty - https://boosty.to/sg_161222</b><br>

<b>Please read this!</b><br>
For version 2.0 it is recommended to use with VAE (to improve generation quality and get rid of blue artifacts): https://huggingface.co/stabilityai/sd-vae-ft-mse-original<br>

<hr/>

<b>I use this template to get good generation results:

Prompt:</b>
RAW photo, *subject*, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3

<b>Example:</b> RAW photo, a close up portrait photo of 26 y.o woman in wastelander clothes, long haircut, pale skin, slim body, background is city ruins, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3


<b>Negative Prompt:</b>
(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck<br>

<b>OR</b><br>

(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation

<b>Euler A or DPM++ 2M Karras with 25 steps<br>
CFG Scale 3,5 - 7<br>
Hires. fix with Latent upscaler<br>
0 Hires steps and Denoising strength 0.25-0.45<br>
Upscale by 1.1-2.0</b>



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/README.md) (3.2 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/feature_extractor/preprocessor_config.json) (518.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/model_index.json) (649.0 B)

- [safety_checker/config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/safety_checker/config.json) (726.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/scheduler/scheduler_config.json) (377.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/text_encoder/config.json) (686.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/tokenizer/tokenizer_config.json) (840.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/unet/config.json) (1.4 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/vae/config.json) (827.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V2.0/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)