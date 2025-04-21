
# blip2-opt-6.7b-coco
---


## README([From Huggingface](https://huggingface.co/Salesforce/blip2-opt-6.7b-coco))

---
language: en
license: mit
tags:
- vision
- image-to-text
- image-captioning
- visual-question-answering
pipeline_tag: image-text-to-text
---

# BLIP-2, OPT-6.7b, fine-tuned on COCO

BLIP-2 model, leveraging [OPT-6.7b](https://huggingface.co/facebook/opt-6.7b) (a large language model with 6.7 billion parameters).
It was introduced in the paper [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) by Li et al. and first released in [this repository](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

Disclaimer: The team releasing BLIP-2 did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

BLIP-2 consists of 3 models: a CLIP-like image encoder, a Querying Transformer (Q-Former) and a large language model.

The authors initialize the weights of the image encoder and large language model from pre-trained checkpoints and keep them frozen
while training the Querying Transformer, which is a BERT-like Transformer encoder that maps a set of "query tokens" to query embeddings,
which bridge the gap between the embedding space of the image encoder and the large language model.

The goal for the model is simply to predict the next text token, giving the query embeddings and the previous text.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/blip2_architecture.jpg"
alt="drawing" width="600"/> 

This allows the model to be used for tasks like:

- image captioning
- visual question answering (VQA)
- chat-like conversations by feeding the image and the previous conversation as prompt to the model

## Direct Use and Downstream Use

You can use the raw model for conditional text generation given an image and optional text. See the [model hub](https://huggingface.co/models?search=Salesforce/blip) to look for
fine-tuned versions on a task that interests you.

## Bias, Risks, Limitations, and Ethical Considerations

BLIP2-OPT uses off-the-shelf OPT as the language model. It inherits the same risks and limitations as mentioned in Meta's model card.

> Like other large language models for which the diversity (or lack thereof) of training
> data induces downstream impact on the quality of our model, OPT-175B has limitations in terms
> of bias and safety. OPT-175B can also have quality issues in terms of generation diversity and
> hallucination. In general, OPT-175B is not immune from the plethora of issues that plague modern
> large language models.
> 
BLIP2 is fine-tuned on image-text datasets (e.g. [LAION](https://laion.ai/blog/laion-400-open-dataset/) ) collected from the internet.  As a result the model itself is potentially vulnerable to generating equivalently inappropriate content or replicating inherent biases in the underlying data.

BLIP2 has not been tested in real world applications. It should not be directly deployed in any applications. Researchers should first carefully assess the safety and fairness of the model in relation to the specific context they’re being deployed within.

## Ethical Considerations
This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.

### How to use

For code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration.forward.example).



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/README.md) (3.9 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/config.json) (6.8 KB)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/merges.txt) (445.7 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/model_state.pdparams) (29.7 GB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/preprocessor_config.json) (432.0 B)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/special_tokens_map.json) (548.0 B)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/tokenizer.json) (2.0 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/tokenizer_config.json) (903.0 B)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-opt-6.7b-coco/vocab.json) (779.6 KB)


[Back to Main](../../)