# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import pandas as pd
import re
import os
from transformers import AutoTokenizer
from datasets import load_dataset
if __name__ == "__main__":
    # control the input length
    sample_num =  400
    split="validation"
    dataset =  load_dataset("hellaswag", split=split, trust_remote_code=True)
    sample_num = min(sample_num, len(dataset))
    print("Sample num",sample_num )
    prompts = []
    
    for i in range(0,sample_num) :
        item = dataset[i]
        context = item["ctx"].strip()
        endings = [e.strip() for e in item["endings"]]
        label = int(item["label"])

        prompt = f"Context: {context}\n"
        for j, ending in enumerate(endings):
            prompt += f"{chr(65+j)}. {ending}\n"
        prompt += "Answer:"
        prompts.append(prompt)
     
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
            print("Raw response text:", response.text)
            print(response.json()['text'][0])
