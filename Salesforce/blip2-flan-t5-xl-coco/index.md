
# blip2-flan-t5-xl-coco
---


## README([From Huggingface](https://huggingface.co/Salesforce/blip2-flan-t5-xl-coco))

---
language: en
license: mit
tags:
- vision
- image-to-text
- image-captioning
- visual-question-answering
pipeline_tag: image-to-text
inference: false
---

# BLIP-2, Flan T5-xl, fine-tuned on COCO

BLIP-2 model, leveraging [Flan T5-xl](https://huggingface.co/google/flan-t5-xl) (a large language model).
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

BLIP2-FlanT5 uses off-the-shelf Flan-T5 as the language model. It inherits the same risks and limitations from [Flan-T5](https://arxiv.org/pdf/2210.11416.pdf):

> Language models, including Flan-T5, can potentially be used for language generation in a harmful way, according to Rae et al. (2021). Flan-T5 should not be used directly in any application, without a prior assessment of safety and fairness concerns specific to the application.

BLIP2 is fine-tuned on image-text datasets (e.g. [LAION](https://laion.ai/blog/laion-400-open-dataset/) ) collected from the internet.  As a result the model itself is potentially vulnerable to generating equivalently inappropriate content or replicating inherent biases in the underlying data.

BLIP2 has not been tested in real world applications. It should not be directly deployed in any applications. Researchers should first carefully assess the safety and fairness of the model in relation to the specific context they’re being deployed within.

## Ethical Considerations
This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.

### How to use

For code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration.forward.example).



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/README.md) (3.8 KB)

- [added_tokens.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/added_tokens.json) (58.0 B)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/config.json) (7.6 KB)

- [model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/model_state.pdparams) (15.2 GB)

- [preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/preprocessor_config.json) (431.0 B)

- [special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/special_tokens_map.json) (2.2 KB)

- [spiece.model](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/spiece.model) (773.1 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/tokenizer.json) (2.3 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/tokenizer_config.json) (2.5 KB)

- [unigram.json](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xl-coco/unigram.json) (1.7 MB)


[Back to Main](../../)