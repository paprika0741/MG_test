# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate"""
import os
import sys
import json
import torch.distributed as dist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import os
import sys
from argparse import Namespace
from contextlib import nullcontext
from typing import Union
 
import torch

import megatron
from megatron.core.models.gpt import GPTModel
from megatron.training import get_model
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import import_module
 
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.checkpointing import save_checkpoint
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from argparse import Namespace

from megatron.core import mpu
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from pretrain_gpt import model_provider
if __name__ == "__main__":
    initialize_megatron(
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'exit_on_missing_checkpoint': True,
        },
    )

    args = get_args()
    model = model_provider(True, True).to(args.params_dtype)
   
    print(args.moe_layer_freq)
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text " "generation.")
    args.exit_on_missing_checkpoint = True

    # Set up model and load checkpoint
    load_context = nullcontext()
    if args.fp8:
        from transformer_engine.pytorch.fp8 import fp8_model_init

        load_context = fp8_model_init()
    with load_context:
        model = get_model(model_provider, wrap_with_ddp=False)
     
    assert len(model) == 1, "Above condition should have caught this"
    if  mpu.get_expert_model_parallel_rank() == 0:
        print(len(model))
        print( model[0].module)
        # for name, param in model[0].module.named_parameters():
        #     print(name, param.shape)
    ep_rank = mpu.get_expert_model_parallel_rank()
    model_dict = state_dict = model[0].module.state_dict()
    rerun_state_machine =  dict()
    # according to the checkpoint of mixtral
    state = {
        "model": model_dict,
        "checkpoint_version": 3.0,
        "iteration":1,
        "args": args,
        "num_floating_point_operations_so_far":0,
        "rerun_state_machine" :rerun_state_machine
    }
    save_dir = args.save
    path = os.path.join(save_dir, f"iter_0000001/mp_rank_00_00{ep_rank}", "model_optim_rng.pt")
    print("save empty weight")
    print("save to ",path)
    print("keys",state.keys() )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    dist.barrier()
    if ep_rank == 0:
        print("All ranks have finished saving.")
        
    # create file latest_checkpointed_iteration.txt
    tracker_file = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
    with open(tracker_file, "w") as f:
        f.write("1")

    