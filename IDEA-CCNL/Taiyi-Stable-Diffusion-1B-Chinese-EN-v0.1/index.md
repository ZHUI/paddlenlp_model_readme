
# Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1
---


## README([From Huggingface](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1))



# Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1

- Main Page:[Fengshenbang](https://fengshenbang-lm.com/)
- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)

# Gradio

We support a [Gradio](https://github.com/gradio-app/gradio) Web UI to run Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1:
[![![Open In Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/IDEA-CCNL/Taiyi-Stable-Diffusion-Chinese)

## 简介 Brief Introduction

首个开源的中英双语Stable Diffusion模型，基于0.2亿筛选过的中文图文对训练。

The first open source Chinese&English Bilingual Stable diffusion, which was trained on 20M filtered Chinese image-text pairs.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 多模态 Multimodal | 太乙 Taiyi | Stable Diffusion |    1B    |     Chinese and English     |

## 模型信息 Model Information

我们将[Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/)数据集(100M)和[Zero](https://zero.so.com/)数据集(23M)用作预训练的数据集，先用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)对这两个数据集的图文对相似性进行打分，取CLIP Score大于0.2的图文对作为我们的训练集。 我们使用[stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)([论文](https://arxiv.org/abs/2112.10752))模型进行继续训练，其中训练分为两个stage。

第一个stage中冻住模型的其他部分，只训练text encoder，以便保留原始模型的生成能力且实现中文概念的对齐。

第二个stage中将全部模型解冻，一起训练text encoder和diffusion model，以便diffusion model更好的适配中文guidance。

第一个stage我们训练了80小时，第二个stage训练了100小时，两个stage都是用了8 x A100。该版本是一个初步的版本，我们将持续优化模型并开源，欢迎交流！

We use [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/)(100M) 和 [Zero](https://zero.so.com/)(23M) as our dataset, and take the image and text pairs with CLIP Score (based on [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)) greater than 0.2 as our Training set. We finetune the [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)([paper](https://arxiv.org/abs/2112.10752)) model for two stage. 

Stage 1: To keep the powerful generative capability of stable diffusion and align Chinese concepts with the images, We only train the text encoder and freeze other part of the model in the first stage. 

Stage 2: We unfreeze both the text encoder and the diffusion model, therefore the diffusion model can have a better compatibility for the Chinese language guidance. 

It takes 80 hours to train the first stage, 100 hours to train the second stage, both stages are based on 8 x A100. This model is a preliminary version and we will update this model continuously and open sourse. Welcome to exchange！

### Result

小桥流水人家，Van Gogh style。
![![](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/resolve/main/result_examples/xiaoqiao_vangogh.png)

小桥流水人家，水彩。
![![](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/resolve/main/result_examples/xiaoqiao_oil_painting.png)

吃过桥米线的猫。
![![](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/resolve/main/result_examples/cat_eating_guoqiao_noodle.png)

穿着宇航服的哈士奇。
![![](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/resolve/main/result_examples/huskiy_wearing_space_suit.png)
## 使用 Usage

### 全精度 Full precision

```py
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1")

prompt = '小桥流水人家，Van Gogh style'
image = pipe(prompt, guidance_scale=10).images[0]  
image.save("小桥.png")
```

### 半精度 Half precision FP16 (CUDA)

添加 `dtype=paddle.float16` 和 `device_map="auto"` 可以快速加载 FP16 的权重，以加快推理速度。
更多信息见 [the optimization docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16#half-precision-weights)。

```py
# !pip install git+https://github.com/huggingface/accelerate
from diffusers import StableDiffusionPipeline
import paddle
torch.backends.cudnn.benchmark = True
pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1", dtype=paddle.float16)
pipe.to('cuda')

prompt = '小桥流水人家，Van Gogh style'
image = pipe(prompt, guidance_scale=10.0).images[0]  
image.save("小桥.png")
```


### 怎样微调 How to finetune

可以参考 refer

https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/finetune_taiyi_stable_diffusion

### webui配置 Configure webui

可以参考 refer

https://github.com/IDEA-CCNL/stable-diffusion-webui/blob/master/README.md

### DreamBooth

https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/stable_diffusion_dreambooth
## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[总论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):

```text
@article{fengshenbang,
  author    = {Jiaxing Zhang and Ruyi Gan and Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

也可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```




## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/README.md) (8.0 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/feature_extractor/preprocessor_config.json) (342.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/model_index.json) (581.0 B)

- [safety_checker/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/safety_checker/model_config.json) (614.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/scheduler/scheduler_config.json) (342.0 B)

- [text_encoder/model_config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/text_encoder/model_config.json) (463.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/tokenizer/added_tokens.json) (2.0 B)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/tokenizer/special_tokens_map.json) (478.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/tokenizer/tokenizer_config.json) (312.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/unet/config.json) (745.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/vae/config.json) (549.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)
