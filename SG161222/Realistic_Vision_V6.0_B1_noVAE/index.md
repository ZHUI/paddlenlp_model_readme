
# Realistic_Vision_V6.0_B1_noVAE
---


## README([From Huggingface](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE))


<strong>Check my exclusive models on Mage: </strong><a href="https://www.mage.space/play/4371756b27bf52e7a1146dc6fe2d969c" rel="noopener noreferrer nofollow"><strong>ParagonXL</strong></a><strong> / </strong><a href="https://www.mage.space/play/df67a9f27f19629a98cb0fb619d1949a" rel="noopener noreferrer nofollow"><strong>NovaXL</strong></a><strong> / </strong><a href="https://www.mage.space/play/d8db06ae964310acb4e090eec03984df" rel="noopener noreferrer nofollow"><strong>NovaXL Lightning</strong></a><strong> / </strong><a href="https://www.mage.space/play/541da1e10976ab82976a5cacc770a413" rel="noopener noreferrer nofollow"><strong>NovaXL V2</strong></a><strong> / </strong><a href="https://www.mage.space/play/a56d2680c464ef25b8c66df126b3f706" rel="noopener noreferrer nofollow"><strong>NovaXL Pony</strong></a><strong> / </strong><a href="https://www.mage.space/play/b0ab6733c3be2408c93523d57a605371" rel="noopener noreferrer nofollow"><strong>NovaXL Pony Lightning</strong></a><strong> / </strong><a href="https://www.mage.space/play/e3b01cd493ed86ed8e4708751b1c9165" rel="noopener noreferrer nofollow"><strong>RealDreamXL</strong></a><strong> / </strong><a href="https://www.mage.space/play/ef062fc389c3f8723002428290c1158c" rel="noopener noreferrer nofollow"><strong>RealDreamXL Lightning</strong></a></p>
<b>This model is available on <a href="https://www.mage.space/">Mage.Space</a> (main sponsor)</b><br>
<b>You can support me directly on Boosty - https://boosty.to/sg_161222</b><br>

<b>Please read this!</b><br>
This is not yet the full version of the model (read the <b>"Model Description"</b> section).<br>
For version 6.0 it is recommended to use with VAE (to improve generation quality and get rid of artifacts): https://huggingface.co/stabilityai/sd-vae-ft-mse-original<br>

<b>Model Description</b><br>
Realistic Vision V6.0 "New Vision" is a global update for the Realistic Vision model, which will be released gradually in several beta versions until the full release. The model is aimed at realism and photorealism.<br>
CivitAI Page: https://civitai.com/models/4201/realistic-vision-v60-b1?modelVersionId=245598

<b>Resolutions (use lower resolution if you get a lot of mutations and stuff like that)</b><br>
- Face Portrait: 896x896<br>
- Portrait: 896x896, 768x1024<br>
- Half Body: 768x1024, 640x1152<br>
- Full Body: 896x896, 768x1024, 640x1152, 1024x768, 1152x640<br>

<b>Improvements</b>
- increased generation resolution to such resolutions as: 896x896, 768x1024, 640x1152, 1024x768, 1152x640. (note. in some cases there may still be mutations, duplications, etc -> will be fixed in future versions).<br>
- improved sfw and nsfw for female and female anatomy (note. not all poses work correctly in such large resolutions -> will be fixed in future versions).<br>

<b>Recommended Workflow</b><br>
Images can be generated with or without Hires.Fix, but it will help improve the generation quality significantly. In some cases it is strictly recommended to use Hires.Fix, namely when generating full body and half body images (note: you can also use Restore Faces or ADetailer).<br>

<b>Recommended Generation Parameters</b><br>
Sampler: DPM++ SDE Karras (25+ steps) / DPM++ 2M SDE (50+ steps)<br>
Negative Prompt: (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck<br>

<b>Recommended Hires.Fix Parameters</b><br>
Sampler: DPM++ SDE Karras or DPM++ 2M SDE<br>
Denoising steps: 10+ (DPM++ SDE Karras) / 20+ (DPM++ 2M SDE (notice. the lower the value of hires steps at a given sampler, the stronger the skin texture and the higher the chance of getting artifacts))<br>
Denoising strength: 0.1-0.3<br>
Upscaler: 4x-UltraSharp / 4x_NMKD-Superscale-SP_178000_G or another<br>
Upscale by: 1.1-2.0+<br>




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/README.md) (4.1 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/config.json) (1.8 KB)

- [diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/diffusion_paddle_model.safetensors) (3.2 GB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/feature_extractor/preprocessor_config.json) (520.0 B)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/merges.txt) (512.3 KB)

- [model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/model.safetensors) (469.5 MB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/model_index.json) (707.0 B)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/preprocessor_config.json) (520.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/scheduler/scheduler_config.json) (376.0 B)

- [scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/scheduler_config.json) (376.0 B)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/special_tokens_map.json) (389.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/text_encoder/config.json) (584.0 B)

- [text_encoder/model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/text_encoder/model.safetensors) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/tokenizer/tokenizer_config.json) (765.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/tokenizer/vocab.json) (1.0 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/tokenizer_config.json) (765.0 B)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/unet/config.json) (1.8 KB)

- [unet/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/unet/diffusion_paddle_model.safetensors) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/vae/config.json) (753.0 B)

- [vae/diffusion_paddle_model.safetensors](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/vae/diffusion_paddle_model.safetensors) (319.1 MB)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/SG161222/Realistic_Vision_V6.0_B1_noVAE/vocab.json) (1.0 MB)


[Back to Main](../../)