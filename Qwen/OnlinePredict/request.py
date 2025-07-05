# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import pandas as pd
import re
import random
import time
import os
from transformers import AutoTokenizer
from datasets import load_dataset
def build_prompt(conversation,tokenizer ):
    prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
    )
    return prompt

if __name__ == "__main__":
    # control the input length
    dataset = load_dataset( "lmsys/lmsys-chat-1m", split="train")
    sample_num = 1000
    random.seed(42) 
    sample_indices = random.sample(range(len(dataset)),sample_num)
    print(sample_indices)
    samples = [dataset[i] for i in sample_indices]
    tokenizer = AutoTokenizer.from_pretrained("/home/download/models/Qwen1.5-MoE-A2.7B-Chat")
    prompts = [ build_prompt(i["conversation"],tokenizer) for  i in  samples ]

     
    filtered_prompts = []
    len_list = []
    for i in prompts:
        tokens = tokenizer.encode(i)
        token_count = len(tokens)
        print(token_count)
        if token_count < 4096:
            len_list.append(token_count)
            filtered_prompts.append(i)
        if len(filtered_prompts) == 500:
            break
    print(len_list)
    
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    for i in range(500):
        print(f"======================{i}==================")
        data = {"prompts":[filtered_prompts[i]], "tokens_to_generate": 1, }
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            print(response.json()['text'][0])
           