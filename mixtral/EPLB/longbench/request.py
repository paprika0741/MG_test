# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import pandas as pd
import re
import os
from transformers import AutoTokenizer
from datasets import load_dataset
def get_longbench_qa(task_name="musique", min_tokens=1000, max_tokens=20000, sample_limit=500):
    tokenizer = AutoTokenizer.from_pretrained("/home/ec2-user/CodeSpace/download/models/Mixtral-8x7B-v0.1")
    print(f"Loading LongBench task: {task_name}")
    dataset = load_dataset("THUDM/LongBench", task_name, split="test")
    print("Filtering samples based on context token length...")
    dataset = dataset.filter(lambda x: min_tokens <= len(tokenizer.encode(x["context"])) <= max_tokens)
    sample_num = min(sample_limit, len(dataset))
    print(f"Sample num after filtering: {sample_num}")
    prompts = []
    for i in range(sample_num):
        item = dataset[i]
        context = item["context"].strip()
        question = item["input"].strip()
        answers = item["answers"]

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        prompts.append(prompt)

  

    return prompts
        

if __name__ == "__main__":
    # control the input length
    sample_num =500
    tokenizer = AutoTokenizer.from_pretrained("/home/ec2-user/CodeSpace/download/models/Mixtral-8x7B-v0.1")
    prompts = []
    prompts +=get_longbench_qa(task_name="qasper")
    prompts +=get_longbench_qa(task_name="narrativeqa")
    prompts +=get_longbench_qa(task_name="musique")
    exit(0)
        
     
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
 
    for i in range(0,sample_num) :
        print(f"=================={i}==================")
        data = {"prompts":[prompts[i] ], "tokens_to_generate": 1}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            # print("Raw response text:", response.text)
            # print(response.json()['text'][0])
