
# anything-v5
---


## README([From Huggingface](https://huggingface.co/stablediffusionapi/anything-v5))



# Anything V5 API Inference

![![generated from modelslab.com](https://assets.modelslab.com/generations/d3d3f607-e8c6-4758-903a-17804fb4002b-0.png)
## Get API Key

Get API key from [ModelsLab](https://modelslab.com/), No Payment needed. 

Replace Key in below code, change **model_id**  to "anything-v5"

Coding in PHP/Node/Java etc? Have a look at docs for more code examples: [View docs](https://stablediffusionapi.com/docs)

Model link: [View model](https://stablediffusionapi.com/models/anything-v5)

Credits: [View credits](https://civitai.com/?query=Anything%20V5)

View all models: [View Models](https://stablediffusionapi.com/models)

    import requests  
    import json  
      
    url =  "https://stablediffusionapi.com/api/v3/dreambooth"  
      
    payload = json.dumps({  
    "key":  "",  
    "model_id":  "anything-v5",  
    "prompt":  "actual 8K portrait photo of gareth person, portrait, happy colors, bright eyes, clear eyes, warm smile, smooth soft skin, big dreamy eyes, beautiful intricate colored hair, symmetrical, anime wide eyes, soft lighting, detailed face, by makoto shinkai, stanley artgerm lau, wlop, rossdraws, concept art, digital painting, looking into camera",  
    "negative_prompt":  "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",  
    "width":  "512",  
    "height":  "512",  
    "samples":  "1",  
    "num_inference_steps":  "30",  
    "safety_checker":  "no",  
    "enhance_prompt":  "yes",  
    "seed":  None,  
    "guidance_scale":  7.5,  
    "multi_lingual":  "no",  
    "panorama":  "no",  
    "self_attention":  "no",  
    "upscale":  "no",  
    "embeddings":  "embeddings_model_id",  
    "lora":  "lora_model_id",  
    "webhook":  None,  
    "track_id":  None  
    })  
      
    headers =  {  
    'Content-Type':  'application/json'  
    }  
      
    response = requests.request("POST", url, headers=headers, data=payload)  
      
    print(response.text)

> Use this coupon code to get 25% off **DMGG0RBN** 



## Model Files

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/README.md) (2.3 KB)

- [feature_extractor/preprocessor_config.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/feature_extractor/preprocessor_config.json) (520.0 B)

- [model_index.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/model_index.json) (617.0 B)

- [safety_checker/config.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/safety_checker/config.json) (681.0 B)

- [safety_checker/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/safety_checker/model_state.pdparams) (1.1 GB)

- [scheduler/scheduler_config.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/scheduler/scheduler_config.json) (462.0 B)

- [text_encoder/config.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/text_encoder/config.json) (636.0 B)

- [text_encoder/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/text_encoder/model_state.pdparams) (469.5 MB)

- [tokenizer/merges.txt](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/tokenizer/merges.txt) (512.3 KB)

- [tokenizer/special_tokens_map.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/tokenizer/special_tokens_map.json) (389.0 B)

- [tokenizer/tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/tokenizer/tokenizer_config.json) (829.0 B)

- [tokenizer/vocab.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/tokenizer/vocab.json) (1.0 MB)

- [unet/config.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/unet/config.json) (1.7 KB)

- [unet/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/unet/model_state.pdparams) (3.2 GB)

- [vae/config.json](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/vae/config.json) (793.0 B)

- [vae/model_state.pdparams](https://paddlenlp.bj.bcebos.com/models/community/stablediffusionapi/anything-v5/vae/model_state.pdparams) (319.1 MB)


[Back to Main](../../)