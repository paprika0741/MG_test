# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import pandas as pd
import re
import random
import os
import time
from transformers import AutoTokenizer
from datasets import load_dataset
def build_prompt_v2(conversation):
    prompt = "<｜begin▁of▁sentence｜>\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt += f"User: {turn['content'].strip()}\n\n"
        elif turn["role"] == "assistant":
            prompt += f"Assistant: {turn['content'].strip()}<｜end▁of▁sentence｜>\n"
    prompt += "Assistant:"
    return prompt
if __name__ == "__main__":
    # control the input length
    dataset = load_dataset( "lmsys/lmsys-chat-1m", split="train")
    sample_num = 400
    random.seed(42) 
    sample_indices = random.sample(range(len(dataset)),sample_num)
    print(sample_indices)
    samples = [dataset[i] for i in sample_indices]
    prompts = [ build_prompt_v2(i["conversation"]) for  i in  samples ]
    tokenizer = AutoTokenizer.from_pretrained("/home/download/models/DeepSeek-V2-Lite")
     
    filtered_prompts = []
    len_list = []
    for i in prompts:
        tokens = tokenizer.encode(i)
        token_count = len(tokens)
        print(token_count)
        if token_count < 2048:
            len_list.append(token_count)
            filtered_prompts.append(i)
        if len(filtered_prompts) == 200:
            break
    print(len_list)
    
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    for i in range(200):
        print(f"======================{i}==================")
        data = {"prompts":[filtered_prompts[i+30]], "tokens_to_generate": 40, "stop_token" : tokenizer.eos_token_id}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            print(response.json()['text'][0])
        time.sleep(2)
