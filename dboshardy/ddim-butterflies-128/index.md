
# ddim-butterflies-128
---


## README([From Huggingface](https://huggingface.co/dboshardy/ddim-butterflies-128))

---
language: en
license: apache-2.0
library_name: diffusers
tags: []
datasets: huggan/smithsonian_butterflies_subset
metrics: []
---

<!-- This model card has been generated automatically according to the information the training script had access to. You
should probably proofread and complete it, then remove this comment. -->

# ddim-butterflies-128

## Model description

This diffusion model is trained with the [🤗 Diffusers](https://github.com/huggingface/diffusers) library 
on the `huggan/smithsonian_butterflies_subset` dataset.

## Intended uses & limitations

#### How to use

```python
# TODO: add an example code snippet for running this diffusion pipeline
```

#### Limitations and bias

[TODO: provide examples of latent issues and potential remediations]

## Training data

[TODO: describe the data used to train the model]

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 32
- eval_batch_size: 32
- gradient_accumulation_steps: 1
- optimizer: AdamW with betas=(None, None), weight_decay=None and epsilon=None
- lr_scheduler: None
- lr_warmup_steps: 250
- ema_inv_gamma: None
- ema_inv_gamma: None
- ema_inv_gamma: None
- mixed_precision: fp16

### Training results

📈 [TensorBoard logs](https://huggingface.co/dboshardy/ddim-butterflies-128/tensorboard?#scalars)





## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/dboshardy/ddim-butterflies-128/README.md) (1.3 KB)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/dboshardy/ddim-butterflies-128/model_index.json) (186.0 B)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/dboshardy/ddim-butterflies-128/scheduler/scheduler_config.json) (279.0 B)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/dboshardy/ddim-butterflies-128/unet/config.json) (788.0 B)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/dboshardy/ddim-butterflies-128/unet/model_state.pdparams) (433.7 MB)


[Back to Main](../../)