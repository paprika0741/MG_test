# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import re
import os
from transformers import AutoTokenizer
 
if __name__ == "__main__":
    prompt =  "Notably, the degree of acceleration is influenced by the numerical relationship between the number of experts and GPUs in Expert Parallelism. In Mixtral-8×7B-Instruct, one-to-one deployment maximizes the effectiveness of capacity-aware inference. To use a different security group, choose Select existing security group and choose an existing security group."
    # prompt = "hello"
    # prompt="who are you"
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    data = {"prompts":[prompt  ], "tokens_to_generate": 1}
    response = requests.put(url, data=json.dumps(data), headers=headers)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.json()['message']}")
    else:
        print("Megatron Response: ")
        print(response.json()['text'][0])
