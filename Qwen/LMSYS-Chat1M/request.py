# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests
import pandas as pd
import re
import os
from transformers import AutoTokenizer
def get_prompt_from_csv(min_len = 8192):
    path = "/home/ec2-user/CodeSpace/SLMoE/4_13_real_trace/data/tokens_csvs"
    filenames = [
        # "128_bin.csv", 
        "256_bin.csv", 
        "512_bin.csv",
        "1024_bin.csv", 
        "2048_bin.csv", 
        "4096_bin.csv",
        "8192_bin.csv" ,
        #   "16384_bin.csv"
    ]
    res = []
    for name in filenames:
        full_path = os.path.join(path, name)
        match = re.search(r"(\d+)_bin.csv", full_path)
        if match:
            value = int(match.group(1))
            # print(value)
            if value > min_len:
                print("===========================\n")
                df = pd.read_csv(full_path)
                # print(f"{name}: {len(df)} rows")
                # print(df.columns)  # 检查实际列名
                # print(len(df["prompt"].to_list()))
                res += df["prompt"].to_list()
                # print(df.head(1))
                print("===========================\n")
    return res
if __name__ == "__main__":
    # control the input length
    promts =  get_prompt_from_csv(min_len = 1024  )
 
    url = "localhost:5000"
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}
    for i in range(200):
        print(f"======================{i}==================")
        data = {"prompts":[promts[i]], "tokens_to_generate": 1}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            print("Megatron Response: ")
            print(response.json()['text'][0])
