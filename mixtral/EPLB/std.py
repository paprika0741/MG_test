import re
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
import glob
import torch
before_tensors = []
after_tensors = []

with open("moe_infer_ideal0_skew0_eplb1.log", "r") as f:
    for line in f:
        if "before tensor" in line:
            match = re.search(r"before tensor\((.*?)\)", line)
            if match:
                tensor_str = match.group(1)
                before_tensor = eval(f"torch.tensor({tensor_str})")
                before_tensors.append(before_tensor)

        elif "after tensor" in line:
            match = re.search(r"after tensor\((.*?)\)", line)
            if match:
                tensor_str = match.group(1)
                after_tensor = eval(f"torch.tensor({tensor_str})")
                after_tensors.append(after_tensor)
num = 250 
print(before_tensors[num])
print("Sum:", before_tensors[num].sum().item())
print("before  std:", before_tensors[num].float().std().item())

print(after_tensors[num])
print("Sum:", after_tensors[num].sum().item())
print("after  std:", after_tensors[0].float().std().item())
