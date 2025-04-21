
# Codegeex-st
---


## README([From Huggingface](https://huggingface.co/THUDM/Codegeex-st))

---
support_training: 0
 
license: Apache License 2.0

---

#### Clone with HTTP
在个人中心->模型->我的模型，查询访问令牌。可以通过令牌进行git仓库的使用。
```bash
 git clone http://git.aistudio.baidu.com/16005791/Codegeex-st.git
```



## Codegeex-st
CodeGeex模型的paddle框架适配版本，支持预训练、SFT、Lora
详见finetune.md
### 环境准备
- PaddlePaddle 3.0-beta
- PaddleNLP 3.0.0b3

### 快速开始
代码块
```
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
path = ""

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype="float16")

input_features = tokenizer(""# language: Python\n# write a bubble sort function\n", return_tensors="pd")
outputs = model.generate(**input_features, max_length=128)
print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
```

## 评测
针对humanevalx测试集合进行评测
评测源代码下载地址：https://github.com/THUDM/CodeGeeX2

### 针对paddle进行代码适配

CodeGeeX2-main/evaluation/generation_paddle.py
```python
import os
import zmq
import time
import json
import paddle
import random
import socket
import argparse

from typing import *

import paddle
import paddlenlp
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlenlp.generation import StoppingCriteria
#from paddlenlp.transformers import AutoModel, AutoTokenizer, StoppingCriteria
from utils import Logger, read_dataset, process_extra_prompt, is_code_generation_finished, cleanup_code

logger = Logger(__name__)


def add_code_generation_specific_args(parser):
    group = parser.add_argument_group("Code Generation")
    group.add_argument(
        "--hostfile",
        type=str,
        default="./hostfile",
    )
    group.add_argument(
        "--channel-ip",
        type=str,
        default=None,
        help="IP for ZeroMQ channel",
    )
    group.add_argument(
        "--channel-port",
        type=int,
        default=5555,
        help="Port for ZeroMQ channel",
    )
    group.add_argument(
        "--master-port",
        type=int,
        default=6007,
        help="Port for distributed channel",
    )
    group.add_argument(
        "--model-per-device",
        type=int,
        default=1,
        help="Number of models per device",
    )
    group.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Max sequence length",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p Probability for sampling",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k for sampling",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    group.add_argument(
        "--greedy",
        type=int,
        default=0,
        help="Use greedy decoding instead of sampling",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size for each GPU",
    )
    group.add_argument(
        "--samples-per-problem",
        type=int,
        default=200,
        help="Number of samples to generate for each problem",
    )
    group.add_argument(
        "--gen-node-world-size",
        type=int,
        default=1,
        help="Number of machines to use for generation",
    )
    group.add_argument(
        '--task-name',
        default="generation",
        help='Name of task',
    )
    group.add_argument(
        '--model-name',
        default="codegeex2-6b",
        help='Name of model, support ["codegeex2-6b", "starcoder", "replit-code-v1-3b", "codegen25-7b-multi", "codegen25-7b-mono", "codegen-16B-multi"]',
    )
    group.add_argument(
        '--data-path',
        required=True,
    )
    group.add_argument(
        '--output-path',
        required=True,
    )
    group.add_argument(
        '--log-path',
        default=None,
        help='Path to log output',
    )
    group.add_argument(
        '--model-path',
        required=True,
    )
    group.add_argument(
        '--dataset-type',
        default="humanevalx",
        help='Identify the evaluation dataset [humanevalx]',
    )
    group.add_argument(
        '--language-type',
        default="python",
        help='Identify the type of programming language to generate',
    )
    group.add_argument(
        '--generation-mode',
        default="instruction",
    )


class CodeStoppingCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length` or meet the code generation stopping criteria.
    """

    def __init__(
        self, 
        max_length: int, 
        micro_batch_size: int, 
        tokenizer,
        dataset_type: str, 
        language_type: str, 
        prompt: str,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.language_type = language_type
        self.prompt = prompt
        self.stop_index = [-1 for _ in range(micro_batch_size)]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for i, input_id in enumerate(input_ids):
            if self.stop_index[i] > -1:
                continue
            code = self.tokenizer.decode(input_id)
            code = code[len(self.prompt):]
            if is_code_generation_finished(
                code,
                dataset_type=self.dataset_type,
                language_type=self.language_type) or input_id.shape[-1] >= self.max_length:
                self.stop_index[i] = len(code) + len(self.prompt)
        if all([s != -1 for s in self.stop_index]):
            return True
        
        return False


def run_generation_distributed(args, model, tokenizer):
    logger.info(f"Connecting to tcp://{args.channel_ip}:{args.channel_port}")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.channel_ip}:{args.channel_port}")
    
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(
        args.output_path,
        f"{args.task_name}-t{args.temperature}-topp{args.top_p}-ns{args.samples_per_problem}-rank{args.rank}.jsonl",
    )
    
    def process(obj,model):
        results = []
        prompt = obj["prompt"]
        if args.generation_mode == "instruction":
            inputs = tokenizer(prompt, return_tensors="pd")
            #inputs = tokenizer([prompt] * args.micro_batch_size, return_tensors="pd")
            #inputs = inputs
            outputs = model.generate(**inputs,
                                    max_length=args.max_length,
                                    do_sample=True if not args.greedy else False,
                                    top_p=args.top_p,
                                    top_k=args.top_k)
            for i, output in enumerate(outputs):
                response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                #response = tokenizer.decode(output)
                res = obj.copy()
                res["generation"] = response[len(prompt):].strip()
                results.append(res)
        elif args.generation_mode == "completion":
            inputs = tokenizer(prompt, return_tensors="pd")
            print(inputs)
            print(type(model))
            #inputs = tokenizer([prompt for _ in range(args.micro_batch_size)], return_tensors="pd")
            #inputs = inputs
            stop_criteria = CodeStoppingCriteria(
                max_length=args.max_length,
                micro_batch_size=args.micro_batch_size,
                tokenizer=tokenizer,
                dataset_type=args.dataset_type,
                language_type=args.language_type,
                prompt=prompt)
            outputs = model.generate(**inputs,
                                    max_length=args.max_length,
                                    do_sample=True if not args.greedy else False,
                                    stopping_criteria=stop_criteria,
                                    top_p=args.top_p,
                                    top_k=args.top_k)
            for i, output in enumerate(outputs):
                response = tokenizer.batch_decode(output, skip_special_tokens=True)
                print(response[0])
                #response = tokenizer.decode(output)
                res = obj.copy()
                res["generation_raw"] = response
                res["generation"] = cleanup_code(
                    response[0], 
                    dataset_type=args.dataset_type,
                    language_type=args.language_type)
                results.append(res)
                break
        
        return results
    
    fout = open(output_path, "w", encoding="utf-8")
    while True:
        socket.send_json({"rank": args.rank, "action": "pull"})
        resp = socket.recv_json()
        
        if resp["task_id"] is None:
            break
        print(type(model))
        current_spec = resp["task_id"]
        results = process(current_spec,model)

        for res in results:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()

        socket.send_json(
            {
                "rank"   : args.rank,
                "action" : "success",
                "task_id": current_spec['task_id']
            }
        )
        socket.recv()

        """
        try:
            if resp["task_id"] is None:
                break

            current_spec = resp["task_id"]
            results = process(current_speci,model)
            print(results)
            for res in results:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()

            socket.send_json(
                {
                    "rank"   : args.rank,
                    "action" : "success",
                    "task_id": current_spec['task_id']
                }
            )
            socket.recv()

        except Exception as e:
            logger.error(f"*** (rank={args.rank}) crashed.")
            logger.error(f"    error: {repr(e)}")
            socket.send_json(
                {
                    "rank"   : args.rank,
                    "action" : "fail",
                    "task_id": current_spec['task_id']
                }
            )
            socket.recv()
            continue
        """

def main(args, node_rank: int, local_rank: int, master_port: int, num_devices: int):
    world_size = args.gen_node_world_size * num_devices
    args.rank = num_devices * node_rank + local_rank
    args.world_size = world_size
    logger.info(f"Generating on rank {args.rank} of {args.world_size}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype="float16")
    except Exception as e:
        logger.error(e)
    
    #model = model.eval()
    # Generate samples.
    print("model.eval",type(model))
    run_generation_distributed(args, model, tokenizer)

    logger.info(f"rank={args.rank} worker finished, waiting ...")
    exit(0)


def server(args):
    logger.info(f"[ server ] starting ...")
    entries = read_dataset(args.data_path, dataset_type=args.dataset_type)

    assert args.samples_per_problem % args.micro_batch_size == 0, "samples_per_problem should be divisible by batch_size"

    for entry in entries.values():
        entry["prompt"] = process_extra_prompt(
            entry["prompt"], 
            language_type=args.language_type, 
            dataset_type=args.dataset_type, 
            generation_mode=args.generation_mode,
        )

    res = []
    for entry in entries.values():
        res.extend([entry] * (args.samples_per_problem // args.micro_batch_size))
    random.shuffle(res)
    all_entries = res

    # setup zeromq channel
    logger.info(f"[ server ] starting up on port {args.channel_port}")
    context = zmq.Context()
    logger.info(f"[ server ] creating socket")
    socket = context.socket(zmq.REP)
    logger.info(f"[ server ] binding to port {args.channel_port}")
    socket.bind(f"tcp://*:{args.channel_port}")

    logger.info(
        f"[ server ] loaded {len(entries)} entries, generating {len(entries) * args.samples_per_problem} samples",
    )

    remaining_entries = all_entries.copy()
    running_workers = args.gen_node_world_size
    num_finished = 0

    logger.info(f"[ server ] listening for requests ...")
    start_time = time.perf_counter()
    while True:
        # Wait for next request from client
        msg = socket.recv_json()
        rank = msg["rank"]
        action = msg["action"]

        if action == "pull":
            if len(remaining_entries) == 0:
                socket.send_json({"task_id": None})
                running_workers -= 1
                logger.info(f"[ server ] Shutting down worker {rank}, remaining {running_workers} workers")
                if running_workers == 0 and num_finished == len(all_entries):
                    logger.info(f"[ server ] All workers finished")
                    break
            else:
                entry = remaining_entries.pop()
                time_elapsed = time.perf_counter() - start_time
                logger.info(f"[ server ] Sending entry {entry['task_id']} to worker {rank}")
                remaining = (
                        len(remaining_entries)
                        / (len(all_entries) - len(remaining_entries))
                        * time_elapsed
                )
                time_per_sampple = 0.0 if num_finished == 0 else time_elapsed / num_finished / args.micro_batch_size
                logger.info(
                    f"[ server ] total {len(all_entries)}, assigned {len(all_entries) - len(remaining_entries)}, finished {num_finished}, elapsed {time_elapsed:.4f}, speed {time_per_sampple:.4f}s/sample, remaining {remaining:.4f}",
                )
                socket.send_json({"task_id": entry})
        else:
            if action == "success":
                logger.info(f"[ server ] {msg['task_id']} is finished")
                socket.send_json({"pong": 1})
            else:
                logger.info(f"[ server ] {msg['task_id']} is not finished")
                remaining_entries.append(msg['task_id'])
                socket.send_json({"pong": 1})
                break

            num_finished += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    add_code_generation_specific_args(parser)
    args = parser.parse_args()
    
    if args.log_path is None:
        args.log_path = os.path.join(args.output_path, "generation.log")

    logger.info("start method: " + torch.multiprocessing.get_start_method())
    
    processes = []
    num_devices = 1
    hosts = open(args.hostfile, "r").readlines()
    hosts = [host.strip() for host in hosts]
    master_port = args.master_port

    node_rank = None
    for i in range(len(hosts)):
        if hosts[i] == socket.gethostbyname(socket.gethostname()):
            node_rank = i
            break
    assert (
            node_rank is not None
    ), f"Could not find hostname ({socket.gethostbyname(socket.gethostname())}) in hostlist"


    #main(args, node_rank, 0, master_port, num_devices)
    #import multiprocessing
    # launch server
    if socket.gethostbyname(socket.gethostname()) == hosts[0]:
        server_process = torch.multiprocessing.Process(target=server, args=(args,))
        logger.info(f"Launching server ...")
        server_process.start()
        processes.append(server_process)

    for i in range(num_devices):
        local_rank = i
        logger.info(f"launching local rank {i}")

        p = torch.multiprocessing.Process(
            target=main,
            args=(args, node_rank, local_rank, master_port, num_devices),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

执行评测：
```
bash scripts/run_humanevalx.sh

⚠️请修改paddle的模型地址和执行的python脚本名称
```


## 关于模型
本模型的推理及微调与训练适配是model_state.pdparams权重格式下做的，由于上传文件大小限制，上传的模型文件转换为例safetensors格式。如有问题欢迎咨询。



## Model Files

- [.gitattributes](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/.gitattributes) (2.6 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/README.md) (16.3 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/config.json) (1.1 KB)

- [finetune.md](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/finetune.md) (13.7 KB)

- [generation_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/generation_config.json) (124.0 B)

- [md5.info](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/md5.info) (540.0 B)

- [model-00001-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/model-00001-of-00004.safetensors) (3.6 GB)

- [model-00002-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/model-00002-of-00004.safetensors) (3.6 GB)

- [model-00003-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/model-00003-of-00004.safetensors) (3.6 GB)

- [model-00004-of-00004.safetensors](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/model-00004-of-00004.safetensors) (829.0 MB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/model.safetensors.index.json) (19.8 KB)

- [tokenizer.model](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/tokenizer.model) (994.5 KB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/THUDM/Codegeex-st/tokenizer_config.json) (163.0 B)


[Back to Main](../../)