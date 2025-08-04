import torch
import os
import re
import argparse
parser = argparse.ArgumentParser(description='Modify model checkpoint file paths.')
parser.add_argument('--root_dir', type=str,  help='Root directory containing model files', default="/home/download/models/mg_core/DeepSeek-V2-Lite/mcore-TP1PP1EP4/")
parser.add_argument('--save_dir', type=str, default="/home/download/models/mg_core/DeepSeek-V2-Lite/router",
                    help='Directory to save individual router weights')
args = parser.parse_args()
root_dir = args.root_dir
os.makedirs(args.save_dir, exist_ok=True)
if os.path.exists(args.save_dir) and os.listdir(args.save_dir):
    print(f"⚠️ Directory '{args.save_dir}' already exists and is not empty. Skipping save.")
    print("Files in the directory:")
    for fname in os.listdir(args.save_dir):
        print("  -", fname)
else:
    target_name = 'model_optim_rng.pt'
    if not os.path.exists(root_dir):
        print(f"not exist: {root_dir}")
        exit(0)
    all_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_name in filenames:
            full_path = os.path.join(dirpath, target_name)
            all_paths.append(full_path)
    state  = torch.load(all_paths[0],map_location="cpu", weights_only=False)
    model_state = state["model"]
    for k, v in model_state.items():
        if "router" in k:
            print(f"Saving {k} ...")
            # Sanitize filename
            filename = k.replace(".", "_") + ".pt"
            save_path = os.path.join(args.save_dir, filename)
            torch.save(v, save_path)
    print(f"\n✅ Saved {len([k for k in model_state if 'router' in k])} router weights to {args.save_dir}")
 