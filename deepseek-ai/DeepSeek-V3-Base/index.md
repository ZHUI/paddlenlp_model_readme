
# DeepSeek-V3-Base
---


## README([From Huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base))

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V3-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


<p align="center">
  <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf"><b>Paper Link</b>👁️</a>
</p>


## 1. Introduction

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 
To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. 
Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. 
We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. 
Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models.
Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training.
In addition, its training process is remarkably stable. 
Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. 
<p align="center">
  <img width="80%" src="figures/benchmark.png">
</p>

## 2. Model Summary

---

**Architecture: Innovative Load Balancing Strategy and Training Objective**

- On top of the efficient architecture of DeepSeek-V2, we pioneer an auxiliary-loss-free strategy for load balancing, which minimizes the performance degradation that arises from encouraging load balancing.
-  We investigate a Multi-Token Prediction (MTP) objective and prove it beneficial to model performance. 
    It can also be used for speculative decoding for inference acceleration. 

---

**Pre-Training: Towards Ultimate Training Efficiency**

- We design an FP8 mixed precision training framework and, for the first time, validate the feasibility and effectiveness of FP8 training on an extremely large-scale model.  
- Through co-design of algorithms, frameworks, and hardware, we overcome the communication bottleneck in cross-node MoE training, nearly achieving full computation-communication overlap.  
  This significantly enhances our training efficiency and reduces the training costs, enabling us to further scale up the model size without additional overhead.  
- At an economical cost of only 2.664M H800 GPU hours, we complete the pre-training of DeepSeek-V3 on 14.8T tokens, producing the currently strongest open-source base model. The subsequent training stages after pre-training require only 0.1M GPU hours.

---

**Post-Training: Knowledge Distillation from DeepSeek-R1**

-   We introduce an innovative methodology to distill reasoning capabilities from the long-Chain-of-Thought (CoT) model, specifically from one of the DeepSeek R1 series models, into standard LLMs, particularly DeepSeek-V3. Our pipeline elegantly incorporates the verification and reflection patterns of R1 into DeepSeek-V3 and notably improves its reasoning performance. Meanwhile, we also maintain a control over the output style and length of DeepSeek-V3.

---


## 3. Model Downloads

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| DeepSeek-V3-Base | 671B | 37B | 128K   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)   |
| DeepSeek-V3   | 671B | 37B |  128K   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3)   |

</div>

**NOTE: The total size of DeepSeek-V3 models on HuggingFace is 685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights.**

To ensure optimal performance and flexibility, we have partnered with open-source communities and hardware vendors to provide multiple ways to run the model locally. For step-by-step guidance, check out Section 6: [How_to Run_Locally](#6-how-to-run-locally).

For developers looking to dive deeper, we recommend exploring [README_WEIGHTS.md](./README_WEIGHTS.md) for details on the Main Model weights and the Multi-Token Prediction (MTP) Modules. Please note that MTP support is currently under active development within the community, and we welcome your contributions and feedback.

## 4. Evaluation Results
### Base Model
#### Standard Benchmarks

<div align="center">


|  | Benchmark (Metric) | # Shots | DeepSeek-V2 | Qwen2.5 72B | LLaMA3.1 405B | DeepSeek-V3 |
|---|-------------------|----------|--------|-------------|---------------|---------|
| | Architecture | - | MoE | Dense | Dense | MoE |
| | # Activated Params | - | 21B | 72B | 405B | 37B |
| | # Total Params | - | 236B | 72B | 405B | 671B |
| English | Pile-test (BPB) | - | 0.606 | 0.638 | **0.542** | 0.548 |
| | BBH (EM) | 3-shot | 78.8 | 79.8 | 82.9 | **87.5** |
| | MMLU (Acc.) | 5-shot | 78.4 | 85.0 | 84.4 | **87.1** |
| | MMLU-Redux (Acc.) | 5-shot | 75.6 | 83.2 | 81.3 | **86.2** |
| | MMLU-Pro (Acc.) | 5-shot | 51.4 | 58.3 | 52.8 | **64.4** |
| | DROP (F1) | 3-shot | 80.4 | 80.6 | 86.0 | **89.0** |
| | ARC-Easy (Acc.) | 25-shot | 97.6 | 98.4 | 98.4 | **98.9** |
| | ARC-Challenge (Acc.) | 25-shot | 92.2 | 94.5 | **95.3** | **95.3** |
| | HellaSwag (Acc.) | 10-shot | 87.1 | 84.8 | **89.2** | 88.9 |
| | PIQA (Acc.) | 0-shot | 83.9 | 82.6 | **85.9** | 84.7 |
| | WinoGrande (Acc.) | 5-shot | **86.3** | 82.3 | 85.2 | 84.9 |
| | RACE-Middle (Acc.) | 5-shot | 73.1 | 68.1 | **74.2** | 67.1 |
| | RACE-High (Acc.) | 5-shot | 52.6 | 50.3 | **56.8** | 51.3 |
| | TriviaQA (EM) | 5-shot | 80.0 | 71.9 | **82.7** | **82.9** |
| | NaturalQuestions (EM) | 5-shot | 38.6 | 33.2 | **41.5** | 40.0 |
| | AGIEval (Acc.) | 0-shot | 57.5 | 75.8 | 60.6 | **79.6** |
| Code | HumanEval (Pass@1) | 0-shot | 43.3 | 53.0 | 54.9 | **65.2** |
| | MBPP (Pass@1) | 3-shot | 65.0 | 72.6 | 68.4 | **75.4** |
| | LiveCodeBench-Base (Pass@1) | 3-shot | 11.6 | 12.9 | 15.5 | **19.4** |
| | CRUXEval-I (Acc.) | 2-shot | 52.5 | 59.1 | 58.5 | **67.3** |
| | CRUXEval-O (Acc.) | 2-shot | 49.8 | 59.9 | 59.9 | **69.8** |
| Math | GSM8K (EM) | 8-shot | 81.6 | 88.3 | 83.5 | **89.3** |
| | MATH (EM) | 4-shot | 43.4 | 54.4 | 49.0 | **61.6** |
| | MGSM (EM) | 8-shot | 63.6 | 76.2 | 69.9 | **79.8** |
| | CMath (EM) | 3-shot | 78.7 | 84.5 | 77.3 | **90.7** |
| Chinese | CLUEWSC (EM) | 5-shot | 82.0 | 82.5 | **83.0** | 82.7 |
| | C-Eval (Acc.) | 5-shot | 81.4 | 89.2 | 72.5 | **90.1** |
| | CMMLU (Acc.) | 5-shot | 84.0 | **89.5** | 73.7 | 88.8 |
| | CMRC (EM) | 1-shot | **77.4** | 75.8 | 76.0 | 76.3 |
| | C3 (Acc.) | 0-shot | 77.4 | 76.7 | **79.7** | 78.6 |
| | CCPM (Acc.) | 0-shot | **93.0** | 88.5 | 78.6 | 92.0 |
| Multilingual | MMMLU-non-English (Acc.) | 5-shot | 64.0 | 74.8 | 73.8 | **79.4** |

</div>

Note: Best results are shown in bold. Scores with a gap not exceeding 0.3 are considered to be at the same level. DeepSeek-V3 achieves the best performance on most benchmarks, especially on math and code tasks.
For more evaluation details, please check our paper. 

#### Context Window
<p align="center">
  <img width="80%" src="figures/niah.png">
</p>

Evaluation results on the ``Needle In A Haystack`` (NIAH) tests.  DeepSeek-V3 performs well across all context window lengths up to **128K**. 

### Chat Model
#### Standard Benchmarks (Models larger than 67B)
<div align="center">

| | **Benchmark (Metric)** | **DeepSeek V2-0506** | **DeepSeek V2.5-0905** | **Qwen2.5 72B-Inst.** | **Llama3.1 405B-Inst.** | **Claude-3.5-Sonnet-1022** | **GPT-4o 0513** | **DeepSeek V3** |
|---|---------------------|---------------------|----------------------|---------------------|----------------------|---------------------------|----------------|----------------|
| | Architecture | MoE | MoE | Dense | Dense | - | - | MoE |
| | # Activated Params | 21B | 21B | 72B | 405B | - | - | 37B |
| | # Total Params | 236B | 236B | 72B | 405B | - | - | 671B |
| English | MMLU (EM) | 78.2 | 80.6 | 85.3 | **88.6** | **88.3** | 87.2 | **88.5** |
| | MMLU-Redux (EM) | 77.9 | 80.3 | 85.6 | 86.2 | **88.9** | 88.0 | **89.1** |
| | MMLU-Pro (EM) | 58.5 | 66.2 | 71.6 | 73.3 | **78.0** | 72.6 | 75.9 |
| | DROP (3-shot F1) | 83.0 | 87.8 | 76.7 | 88.7 | 88.3 | 83.7 | **91.6** |
| | IF-Eval (Prompt Strict) | 57.7 | 80.6 | 84.1 | 86.0 | **86.5** | 84.3 | 86.1 |
| | GPQA-Diamond (Pass@1) | 35.3 | 41.3 | 49.0 | 51.1 | **65.0** | 49.9 | 59.1 |
| | SimpleQA (Correct) | 9.0 | 10.2 | 9.1 | 17.1 | 28.4 | **38.2** | 24.9 |
| | FRAMES (Acc.) | 66.9 | 65.4 | 69.8 | 70.0 | 72.5 | **80.5** | 73.3 |
| | LongBench v2 (Acc.) | 31.6 | 35.4 | 39.4 | 36.1 | 41.0 | 48.1 | **48.7** |
| Code | HumanEval-Mul (Pass@1) | 69.3 | 77.4 | 77.3 | 77.2 | 81.7 | 80.5 | **82.6** |
| | LiveCodeBench (Pass@1-COT) | 18.8 | 29.2 | 31.1 | 28.4 | 36.3 | 33.4 | **40.5** |
| | LiveCodeBench (Pass@1) | 20.3 | 28.4 | 28.7 | 30.1 | 32.8 | 34.2 | **37.6** |
| | Codeforces (Percentile) | 17.5 | 35.6 | 24.8 | 25.3 | 20.3 | 23.6 | **51.6** |
| | SWE Verified (Resolved) | - | 22.6 | 23.8 | 24.5 | **50.8** | 38.8 | 42.0 |
| | Aider-Edit (Acc.) | 60.3 | 71.6 | 65.4 | 63.9 | **84.2** | 72.9 | 79.7 |
| | Aider-Polyglot (Acc.) | - | 18.2 | 7.6 | 5.8 | 45.3 | 16.0 | **49.6** |
| Math | AIME 2024 (Pass@1) | 4.6 | 16.7 | 23.3 | 23.3 | 16.0 | 9.3 | **39.2** |
| | MATH-500 (EM) | 56.3 | 74.7 | 80.0 | 73.8 | 78.3 | 74.6 | **90.2** |
| | CNMO 2024 (Pass@1) | 2.8 | 10.8 | 15.9 | 6.8 | 13.1 | 10.8 | **43.2** |
| Chinese | CLUEWSC (EM) | 89.9 | 90.4 | **91.4** | 84.7 | 85.4 | 87.9 | 90.9 |
| | C-Eval (EM) | 78.6 | 79.5 | 86.1 | 61.5 | 76.7 | 76.0 | **86.5** |
| | C-SimpleQA (Correct) | 48.5 | 54.1 | 48.4 | 50.4 | 51.3 | 59.3 | **64.8** |

Note: All models are evaluated in a configuration that limits the output length to 8K. Benchmarks containing fewer than 1000 samples are tested multiple times using varying temperature settings to derive robust final results. DeepSeek-V3 stands as the best-performing open-source model, and also exhibits competitive performance against frontier closed-source models.

</div>


####  Open Ended Generation Evaluation

<div align="center">



| Model | Arena-Hard | AlpacaEval 2.0 |
|-------|------------|----------------|
| DeepSeek-V2.5-0905 | 76.2 | 50.5 |
| Qwen2.5-72B-Instruct | 81.2 | 49.1 |
| LLaMA-3.1 405B | 69.3 | 40.5 |
| GPT-4o-0513 | 80.4 | 51.1 |
| Claude-Sonnet-3.5-1022 | 85.2 | 52.0 |
| DeepSeek-V3 | **85.5** | **70.0** |

Note: English open-ended conversation evaluations. For AlpacaEval 2.0, we use the length-controlled win rate as the metric.
</div>


## 5. Chat Website & API Platform
You can chat with DeepSeek-V3 on DeepSeek's official website: [chat.deepseek.com](https://chat.deepseek.com/sign_in)

We also provide OpenAI-Compatible API at DeepSeek Platform: [platform.deepseek.com](https://platform.deepseek.com/)

## 6. How to Run Locally

DeepSeek-V3 can be deployed locally using the following hardware and open-source community software:

1. **DeepSeek-Infer Demo**: We provide a simple and lightweight demo for FP8 and BF16 inference.
2. **SGLang**: Fully support the DeepSeek-V3 model in both BF16 and FP8 inference modes.
3. **LMDeploy**: Enables efficient FP8 and BF16 inference for local and cloud deployment.
4. **TensorRT-LLM**: Currently supports BF16 inference and INT4/8 quantization, with FP8 support coming soon.
5. **vLLM**: Support DeekSeek-V3 model with FP8 and BF16 modes for tensor parallelism and pipeline parallelism.
6. **AMD GPU**: Enables running the DeepSeek-V3 model on AMD GPUs via SGLang in both BF16 and FP8 modes.
7. **Huawei Ascend NPU**: Supports running DeepSeek-V3 on Huawei Ascend devices.

Since FP8 training is natively adopted in our framework, we only provide FP8 weights. If you require BF16 weights for experimentation, you can use the provided conversion script to perform the transformation.

Here is an example of converting FP8 weights to BF16:

```shell
cd inference
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8_weights --output-bf16-hf-path /path/to/bf16_weights
```

**NOTE: Huggingface's Transformers has not been directly supported yet.**

### 6.1 Inference with DeepSeek-Infer Demo (example only)

#### Model Weights & Demo Code Preparation

First, clone our DeepSeek-V3 GitHub repository:

```shell
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
```

Navigate to the `inference` folder and install dependencies listed in `requirements.txt`.

```shell
cd DeepSeek-V3/inference
pip install -r requirements.txt
```

Download the model weights from HuggingFace, and put them into `/path/to/DeepSeek-V3` folder.

#### Model Weights Conversion

Convert HuggingFace model weights to a specific format:

```shell
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16
```

#### Run

Then you can chat with DeepSeek-V3:

```shell
torchrun --nnodes 2 --nproc-per-node 8 generate.py --node-rank $RANK --master-addr $ADDR --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200
```

Or batch inference on a given file:

```shell
torchrun --nnodes 2 --nproc-per-node 8 generate.py --node-rank $RANK --master-addr $ADDR --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --input-file $FILE
```

### 6.2 Inference with SGLang (recommended)

[SGLang](https://github.com/sgl-project/sglang) currently supports MLA optimizations, FP8 (W8A8), FP8 KV Cache, and Torch Compile, delivering state-of-the-art latency and throughput performance among open-source frameworks.

Notably, [SGLang v0.4.1](https://github.com/sgl-project/sglang/releases/tag/v0.4.1) fully supports running DeepSeek-V3 on both **NVIDIA and AMD GPUs**, making it a highly versatile and robust solution.

Here are the launch instructions from the SGLang team: https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

### 6.3 Inference with LMDeploy (recommended)
[LMDeploy](https://github.com/InternLM/lmdeploy), a flexible and high-performance inference and serving framework tailored for large language models, now supports DeepSeek-V3. It offers both offline pipeline processing and online deployment capabilities, seamlessly integrating with PyTorch-based workflows.

For comprehensive step-by-step instructions on running DeepSeek-V3 with LMDeploy, please refer to here: https://github.com/InternLM/lmdeploy/issues/2960


### 6.4 Inference with TRT-LLM (recommended)

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) now supports the DeepSeek-V3 model, offering precision options such as BF16 and INT4/INT8 weight-only. Support for FP8 is currently in progress and will be released soon. You can access the custom branch of TRTLLM specifically for DeepSeek-V3 support through the following link to experience the new features directly: https://github.com/NVIDIA/TensorRT-LLM/tree/deepseek/examples/deepseek_v3. 

### 6.5 Inference with vLLM (recommended)

[vLLM](https://github.com/vllm-project/vllm) v0.6.6 supports DeepSeek-V3 inference for FP8 and BF16 modes on both NVIDIA and AMD GPUs. Aside from standard techniques, vLLM offers _pipeline parallelism_ allowing you to run this model on multiple machines connected by networks. For detailed guidance, please refer to the [vLLM instructions](https://docs.vllm.ai/en/latest/serving/distributed_serving.html). Please feel free to follow [the enhancement plan](https://github.com/vllm-project/vllm/issues/11539) as well.

### 6.6 Recommended Inference Functionality with AMD GPUs

In collaboration with the AMD team, we have achieved Day-One support for AMD GPUs using SGLang, with full compatibility for both FP8 and BF16 precision. For detailed guidance, please refer to the [SGLang instructions](#63-inference-with-lmdeploy-recommended).

### 6.7 Recommended Inference Functionality with Huawei Ascend NPUs
The [MindIE](https://www.hiascend.com/en/software/mindie) framework from the Huawei Ascend community has successfully adapted the BF16 version of DeepSeek-V3. For step-by-step guidance on Ascend NPUs, please follow the [instructions here](https://modelers.cn/models/MindIE/deepseekv3).


## 7. License
This code repository is licensed under [the MIT License](LICENSE-CODE). The use of DeepSeek-V3 Base/Chat models is subject to [the Model License](LICENSE-MODEL). DeepSeek-V3 series (including Base and Chat) supports commercial use.

## 8. Citation
```
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report}, 
      author={DeepSeek-AI and Aixin Liu and Bei Feng and Bing Xue and Bingxuan Wang and Bochao Wu and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenyu Zhang and Chong Ruan and Damai Dai and Daya Guo and Dejian Yang and Deli Chen and Dongjie Ji and Erhang Li and Fangyun Lin and Fucong Dai and Fuli Luo and Guangbo Hao and Guanting Chen and Guowei Li and H. Zhang and Han Bao and Hanwei Xu and Haocheng Wang and Haowei Zhang and Honghui Ding and Huajian Xin and Huazuo Gao and Hui Li and Hui Qu and J. L. Cai and Jian Liang and Jianzhong Guo and Jiaqi Ni and Jiashi Li and Jiawei Wang and Jin Chen and Jingchang Chen and Jingyang Yuan and Junjie Qiu and Junlong Li and Junxiao Song and Kai Dong and Kai Hu and Kaige Gao and Kang Guan and Kexin Huang and Kuai Yu and Lean Wang and Lecong Zhang and Lei Xu and Leyi Xia and Liang Zhao and Litong Wang and Liyue Zhang and Meng Li and Miaojun Wang and Mingchuan Zhang and Minghua Zhang and Minghui Tang and Mingming Li and Ning Tian and Panpan Huang and Peiyi Wang and Peng Zhang and Qiancheng Wang and Qihao Zhu and Qinyu Chen and Qiushi Du and R. J. Chen and R. L. Jin and Ruiqi Ge and Ruisong Zhang and Ruizhe Pan and Runji Wang and Runxin Xu and Ruoyu Zhang and Ruyi Chen and S. S. Li and Shanghao Lu and Shangyan Zhou and Shanhuang Chen and Shaoqing Wu and Shengfeng Ye and Shengfeng Ye and Shirong Ma and Shiyu Wang and Shuang Zhou and Shuiping Yu and Shunfeng Zhou and Shuting Pan and T. Wang and Tao Yun and Tian Pei and Tianyu Sun and W. L. Xiao and Wangding Zeng and Wanjia Zhao and Wei An and Wen Liu and Wenfeng Liang and Wenjun Gao and Wenqin Yu and Wentao Zhang and X. Q. Li and Xiangyue Jin and Xianzu Wang and Xiao Bi and Xiaodong Liu and Xiaohan Wang and Xiaojin Shen and Xiaokang Chen and Xiaokang Zhang and Xiaosha Chen and Xiaotao Nie and Xiaowen Sun and Xiaoxiang Wang and Xin Cheng and Xin Liu and Xin Xie and Xingchao Liu and Xingkai Yu and Xinnan Song and Xinxia Shan and Xinyi Zhou and Xinyu Yang and Xinyuan Li and Xuecheng Su and Xuheng Lin and Y. K. Li and Y. Q. Wang and Y. X. Wei and Y. X. Zhu and Yang Zhang and Yanhong Xu and Yanhong Xu and Yanping Huang and Yao Li and Yao Zhao and Yaofeng Sun and Yaohui Li and Yaohui Wang and Yi Yu and Yi Zheng and Yichao Zhang and Yifan Shi and Yiliang Xiong and Ying He and Ying Tang and Yishi Piao and Yisong Wang and Yixuan Tan and Yiyang Ma and Yiyuan Liu and Yongqiang Guo and Yu Wu and Yuan Ou and Yuchen Zhu and Yuduan Wang and Yue Gong and Yuheng Zou and Yujia He and Yukun Zha and Yunfan Xiong and Yunxian Ma and Yuting Yan and Yuxiang Luo and Yuxiang You and Yuxuan Liu and Yuyang Zhou and Z. F. Wu and Z. Z. Ren and Zehui Ren and Zhangli Sha and Zhe Fu and Zhean Xu and Zhen Huang and Zhen Zhang and Zhenda Xie and Zhengyan Zhang and Zhewen Hao and Zhibin Gou and Zhicheng Ma and Zhigang Yan and Zhihong Shao and Zhipeng Xu and Zhiyu Wu and Zhongyu Zhang and Zhuoshu Li and Zihui Gu and Zijia Zhu and Zijun Liu and Zilin Li and Ziwei Xie and Ziyang Song and Ziyi Gao and Zizheng Pan},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437}, 
}
```

## 9. Contact
If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).




## Model Files

- [LICENSE-CODE](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/LICENSE-CODE) (1.0 KB)

- [LICENSE-MODEL](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/LICENSE-MODEL) (13.4 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/README.md) (22.1 KB)

- [README_WEIGHTS.md](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/README_WEIGHTS.md) (3.6 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/config.json) (0.0 B)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/configuration.json) (73.0 B)

- [configuration_deepseek.py](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/configuration_deepseek.py) (10.3 KB)

- [figures/benchmark.png](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/figures/benchmark.png) (179.3 KB)

- [figures/niah.png](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/figures/niah.png) (105.9 KB)

- [inference/__pycache__/kernel.cpython-310.pyc](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/__pycache__/kernel.cpython-310.pyc) (4.3 KB)

- [inference/configs/config_16B.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/configs/config_16B.json) (417.0 B)

- [inference/configs/config_236B.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/configs/config_236B.json) (455.0 B)

- [inference/configs/config_671B.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/configs/config_671B.json) (503.0 B)

- [inference/convert.py](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/convert.py) (3.2 KB)

- [inference/fp8_cast_bf16.py](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/fp8_cast_bf16.py) (3.2 KB)

- [inference/generate.py](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/generate.py) (5.3 KB)

- [inference/kernel.py](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/kernel.py) (4.2 KB)

- [inference/model.py](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/model.py) (17.1 KB)

- [inference/requirements.txt](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/inference/requirements.txt) (66.0 B)

- [model-00001-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00001-of-000163.safetensors) (8.0 GB)

- [model-00002-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00002-of-000163.safetensors) (8.0 GB)

- [model-00003-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00003-of-000163.safetensors) (8.0 GB)

- [model-00004-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00004-of-000163.safetensors) (8.0 GB)

- [model-00005-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00005-of-000163.safetensors) (8.0 GB)

- [model-00006-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00006-of-000163.safetensors) (8.1 GB)

- [model-00007-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00007-of-000163.safetensors) (8.0 GB)

- [model-00008-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00008-of-000163.safetensors) (8.0 GB)

- [model-00009-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00009-of-000163.safetensors) (8.0 GB)

- [model-00010-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00010-of-000163.safetensors) (8.0 GB)

- [model-00011-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00011-of-000163.safetensors) (8.0 GB)

- [model-00012-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00012-of-000163.safetensors) (2.5 GB)

- [model-00013-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00013-of-000163.safetensors) (8.0 GB)

- [model-00014-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00014-of-000163.safetensors) (8.0 GB)

- [model-00015-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00015-of-000163.safetensors) (8.0 GB)

- [model-00016-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00016-of-000163.safetensors) (8.0 GB)

- [model-00017-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00017-of-000163.safetensors) (8.0 GB)

- [model-00018-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00018-of-000163.safetensors) (8.0 GB)

- [model-00019-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00019-of-000163.safetensors) (8.0 GB)

- [model-00020-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00020-of-000163.safetensors) (8.0 GB)

- [model-00021-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00021-of-000163.safetensors) (8.0 GB)

- [model-00022-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00022-of-000163.safetensors) (8.0 GB)

- [model-00023-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00023-of-000163.safetensors) (8.0 GB)

- [model-00024-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00024-of-000163.safetensors) (8.0 GB)

- [model-00025-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00025-of-000163.safetensors) (8.0 GB)

- [model-00026-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00026-of-000163.safetensors) (8.0 GB)

- [model-00027-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00027-of-000163.safetensors) (8.0 GB)

- [model-00028-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00028-of-000163.safetensors) (8.0 GB)

- [model-00029-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00029-of-000163.safetensors) (8.0 GB)

- [model-00030-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00030-of-000163.safetensors) (8.0 GB)

- [model-00031-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00031-of-000163.safetensors) (8.0 GB)

- [model-00032-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00032-of-000163.safetensors) (8.0 GB)

- [model-00033-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00033-of-000163.safetensors) (8.0 GB)

- [model-00034-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00034-of-000163.safetensors) (3.3 GB)

- [model-00035-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00035-of-000163.safetensors) (8.0 GB)

- [model-00036-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00036-of-000163.safetensors) (8.0 GB)

- [model-00037-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00037-of-000163.safetensors) (8.0 GB)

- [model-00038-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00038-of-000163.safetensors) (8.0 GB)

- [model-00039-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00039-of-000163.safetensors) (8.0 GB)

- [model-00040-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00040-of-000163.safetensors) (8.0 GB)

- [model-00041-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00041-of-000163.safetensors) (8.0 GB)

- [model-00042-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00042-of-000163.safetensors) (8.0 GB)

- [model-00043-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00043-of-000163.safetensors) (8.0 GB)

- [model-00044-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00044-of-000163.safetensors) (8.0 GB)

- [model-00045-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00045-of-000163.safetensors) (8.0 GB)

- [model-00046-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00046-of-000163.safetensors) (8.0 GB)

- [model-00047-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00047-of-000163.safetensors) (8.0 GB)

- [model-00048-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00048-of-000163.safetensors) (8.0 GB)

- [model-00049-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00049-of-000163.safetensors) (8.0 GB)

- [model-00050-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00050-of-000163.safetensors) (8.0 GB)

- [model-00051-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00051-of-000163.safetensors) (8.0 GB)

- [model-00052-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00052-of-000163.safetensors) (8.0 GB)

- [model-00053-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00053-of-000163.safetensors) (8.0 GB)

- [model-00054-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00054-of-000163.safetensors) (8.0 GB)

- [model-00055-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00055-of-000163.safetensors) (8.0 GB)

- [model-00056-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00056-of-000163.safetensors) (3.3 GB)

- [model-00057-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00057-of-000163.safetensors) (8.0 GB)

- [model-00058-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00058-of-000163.safetensors) (8.0 GB)

- [model-00059-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00059-of-000163.safetensors) (8.0 GB)

- [model-00060-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00060-of-000163.safetensors) (8.0 GB)

- [model-00061-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00061-of-000163.safetensors) (8.0 GB)

- [model-00062-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00062-of-000163.safetensors) (8.0 GB)

- [model-00063-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00063-of-000163.safetensors) (8.0 GB)

- [model-00064-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00064-of-000163.safetensors) (8.0 GB)

- [model-00065-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00065-of-000163.safetensors) (8.0 GB)

- [model-00066-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00066-of-000163.safetensors) (8.0 GB)

- [model-00067-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00067-of-000163.safetensors) (8.0 GB)

- [model-00068-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00068-of-000163.safetensors) (8.0 GB)

- [model-00069-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00069-of-000163.safetensors) (8.0 GB)

- [model-00070-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00070-of-000163.safetensors) (8.0 GB)

- [model-00071-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00071-of-000163.safetensors) (8.0 GB)

- [model-00072-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00072-of-000163.safetensors) (8.0 GB)

- [model-00073-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00073-of-000163.safetensors) (8.0 GB)

- [model-00074-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00074-of-000163.safetensors) (8.0 GB)

- [model-00075-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00075-of-000163.safetensors) (8.0 GB)

- [model-00076-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00076-of-000163.safetensors) (8.0 GB)

- [model-00077-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00077-of-000163.safetensors) (8.0 GB)

- [model-00078-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00078-of-000163.safetensors) (3.3 GB)

- [model-00079-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00079-of-000163.safetensors) (8.0 GB)

- [model-00080-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00080-of-000163.safetensors) (8.0 GB)

- [model-00081-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00081-of-000163.safetensors) (8.0 GB)

- [model-00082-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00082-of-000163.safetensors) (8.0 GB)

- [model-00083-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00083-of-000163.safetensors) (8.0 GB)

- [model-00084-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00084-of-000163.safetensors) (8.0 GB)

- [model-00085-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00085-of-000163.safetensors) (8.0 GB)

- [model-00086-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00086-of-000163.safetensors) (8.0 GB)

- [model-00087-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00087-of-000163.safetensors) (8.0 GB)

- [model-00088-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00088-of-000163.safetensors) (8.0 GB)

- [model-00089-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00089-of-000163.safetensors) (8.0 GB)

- [model-00090-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00090-of-000163.safetensors) (8.0 GB)

- [model-00091-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00091-of-000163.safetensors) (8.0 GB)

- [model-00092-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00092-of-000163.safetensors) (8.0 GB)

- [model-00093-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00093-of-000163.safetensors) (8.0 GB)

- [model-00094-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00094-of-000163.safetensors) (8.0 GB)

- [model-00095-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00095-of-000163.safetensors) (8.0 GB)

- [model-00096-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00096-of-000163.safetensors) (8.0 GB)

- [model-00097-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00097-of-000163.safetensors) (8.0 GB)

- [model-00098-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00098-of-000163.safetensors) (8.0 GB)

- [model-00099-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00099-of-000163.safetensors) (8.0 GB)

- [model-00100-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00100-of-000163.safetensors) (3.3 GB)

- [model-00101-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00101-of-000163.safetensors) (8.0 GB)

- [model-00102-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00102-of-000163.safetensors) (8.0 GB)

- [model-00103-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00103-of-000163.safetensors) (8.0 GB)

- [model-00104-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00104-of-000163.safetensors) (8.0 GB)

- [model-00105-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00105-of-000163.safetensors) (8.0 GB)

- [model-00106-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00106-of-000163.safetensors) (8.0 GB)

- [model-00107-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00107-of-000163.safetensors) (8.0 GB)

- [model-00108-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00108-of-000163.safetensors) (8.0 GB)

- [model-00109-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00109-of-000163.safetensors) (8.0 GB)

- [model-00110-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00110-of-000163.safetensors) (8.0 GB)

- [model-00111-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00111-of-000163.safetensors) (8.0 GB)

- [model-00112-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00112-of-000163.safetensors) (8.0 GB)

- [model-00113-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00113-of-000163.safetensors) (8.0 GB)

- [model-00114-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00114-of-000163.safetensors) (8.0 GB)

- [model-00115-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00115-of-000163.safetensors) (8.0 GB)

- [model-00116-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00116-of-000163.safetensors) (8.0 GB)

- [model-00117-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00117-of-000163.safetensors) (8.0 GB)

- [model-00118-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00118-of-000163.safetensors) (8.0 GB)

- [model-00119-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00119-of-000163.safetensors) (8.0 GB)

- [model-00120-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00120-of-000163.safetensors) (8.0 GB)

- [model-00121-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00121-of-000163.safetensors) (8.0 GB)

- [model-00122-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00122-of-000163.safetensors) (3.3 GB)

- [model-00123-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00123-of-000163.safetensors) (8.0 GB)

- [model-00124-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00124-of-000163.safetensors) (8.0 GB)

- [model-00125-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00125-of-000163.safetensors) (8.0 GB)

- [model-00126-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00126-of-000163.safetensors) (8.0 GB)

- [model-00127-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00127-of-000163.safetensors) (8.0 GB)

- [model-00128-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00128-of-000163.safetensors) (8.0 GB)

- [model-00129-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00129-of-000163.safetensors) (8.0 GB)

- [model-00130-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00130-of-000163.safetensors) (8.0 GB)

- [model-00131-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00131-of-000163.safetensors) (8.0 GB)

- [model-00132-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00132-of-000163.safetensors) (8.0 GB)

- [model-00133-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00133-of-000163.safetensors) (8.0 GB)

- [model-00134-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00134-of-000163.safetensors) (8.0 GB)

- [model-00135-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00135-of-000163.safetensors) (8.0 GB)

- [model-00136-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00136-of-000163.safetensors) (8.0 GB)

- [model-00137-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00137-of-000163.safetensors) (8.0 GB)

- [model-00138-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00138-of-000163.safetensors) (8.0 GB)

- [model-00139-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00139-of-000163.safetensors) (8.0 GB)

- [model-00140-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00140-of-000163.safetensors) (8.0 GB)

- [model-00141-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00141-of-000163.safetensors) (5.9 GB)

- [model-00142-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00142-of-000163.safetensors) (8.0 GB)

- [model-00143-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00143-of-000163.safetensors) (8.0 GB)

- [model-00144-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00144-of-000163.safetensors) (8.0 GB)

- [model-00145-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00145-of-000163.safetensors) (8.0 GB)

- [model-00146-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00146-of-000163.safetensors) (8.0 GB)

- [model-00147-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00147-of-000163.safetensors) (8.0 GB)

- [model-00148-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00148-of-000163.safetensors) (8.0 GB)

- [model-00149-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00149-of-000163.safetensors) (8.0 GB)

- [model-00150-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00150-of-000163.safetensors) (8.0 GB)

- [model-00151-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00151-of-000163.safetensors) (8.0 GB)

- [model-00152-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00152-of-000163.safetensors) (8.0 GB)

- [model-00153-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00153-of-000163.safetensors) (8.0 GB)

- [model-00154-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00154-of-000163.safetensors) (8.0 GB)

- [model-00155-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00155-of-000163.safetensors) (8.0 GB)

- [model-00156-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00156-of-000163.safetensors) (8.0 GB)

- [model-00157-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00157-of-000163.safetensors) (8.0 GB)

- [model-00158-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00158-of-000163.safetensors) (8.0 GB)

- [model-00159-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00159-of-000163.safetensors) (8.0 GB)

- [model-00160-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00160-of-000163.safetensors) (8.0 GB)

- [model-00161-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00161-of-000163.safetensors) (8.0 GB)

- [model-00162-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00162-of-000163.safetensors) (8.0 GB)

- [model-00163-of-000163.safetensors](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model-00163-of-000163.safetensors) (8.6 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model.safetensors.index.json) (4.3 MB)

- [model.safetensors.index.json.fp8](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/model.safetensors.index.json.fp8) (8.5 MB)

- [modeling_deepseek.py](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/modeling_deepseek.py) (74.0 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/tokenizer.json) (7.5 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/deepseek-ai/DeepSeek-V3-Base/tokenizer_config.json) (3.1 KB)


[Back to Main](../../)