# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
# from pretrain_gpt import model_provider
import torch
import sys
import time
import tqdm
import random
import warnings
from argparse import Namespace
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from transformers import AutoTokenizer
from megatron.core.transformer.module import MegatronModule

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from datasets import load_dataset

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.core import mpu
import json
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model
import asyncio
from typing import AsyncIterator, List
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from examples.inference.gpt.utils import add_common_inference_args, build_requests
# copy from pretrain_gpt.py
def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )

    return model

# my_arguments.py

# def add_my_inference_args(parser):
#     group = parser.add_argument_group(title='My Custom Inference Args')
#     group.add_argument("--num",
#                       type=int,
#                       default=64)
#     group.add_argument("--bs",
#                       type=int,
#                       default=2)
#     group.add_argument("--input_len",
#                       type=int,
#                       default=32)
#     group.add_argument('--output_len', type=int, default=16,
#                        help='Number of tokens to generate.')
#     return parser


def add_static_inference_args(parser):
    """Static inference arguments."""

    add_common_inference_args(parser)

    group = parser.add_argument_group(title='Static inference')
    group.add_argument(
        "--max-batch-size", type=int, default=8, dest="inference_max_requests",
        help='Max number of prompts to process at once'
    )
    group.add_argument("--stream", action="store_true", default=False, help="Stream output tokens")
    group.add_argument("--output-path", type=str, default='/tmp/output.json', help="Path to save generations as JSON")
    group.add_argument("--dataset", type=str , default="lmsys/lmsys-chat-1m", help="Dataset")
    group.add_argument("--hf_path", type=str , default="/home/download/models/DeepSeek-V2-Lite", help="tokenizer")

    group.add_argument("--num",
                      type=int,
                      default=64)
    group.add_argument("--bs",
                      type=int,
                      default=2)
    group.add_argument("--input_len",
                      type=int,
                      default=32)
    group.add_argument('--output_len', type=int, default=16,
                       help='Number of tokens to generate.')
    return parser
def build_prompt_deepseek(conversation):
    prompt = "<｜begin▁of▁sentence｜>\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt += f"User: {turn['content'].strip()}\n\n"
        elif turn["role"] == "assistant":
            prompt += f"Assistant: {turn['content'].strip()}<｜end▁of▁sentence｜>\n"
    prompt += "Assistant:"
    return prompt
def build_prompt_mixtral(conversation):
    prompt = ""
    for turn in conversation:
        role = "user" if turn["role"] == "user" else "assistant"
        prompt += f"<|{role}|>\n{turn['content'].strip()}\n"
    prompt += "<|assistant|>\n"
    return prompt
def build_prompt_qwen(conversation,tokenizer ):
    prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
    )
    return prompt
def prepare_prompts(args):
    dataset = load_dataset( "lmsys/lmsys-chat-1m", split="train")
    sample_num = args.num
    random.seed(42) 
    sample_indices = random.sample(range(len(dataset)),sample_num)
    print(sample_indices)
    samples = [dataset[i] for i in sample_indices]
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path)
    if "DeepSeek" in args.hf_path:
        prompts = [ build_prompt_deepseek(i["conversation"]) for  i in  samples ]
    elif "Mixtral" in   args.hf_path:
        prompts = [ build_prompt_mixtral(i["conversation"]) for  i in  samples ]
    elif "Qwen1.5" in   args.hf_path:
        prompts = [ build_prompt_qwen(i["conversation"],tokenizer) for  i in  samples ]
    else:
        raise NotImplementedError
    
    filtered_prompts = []
 
    for i in prompts:
        tokens = tokenizer.encode(i)
        truncated_tokens = tokens[: args.input_len]
        print(f"Original token count: { len(tokens)} -> {len(truncated_tokens)}",)
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        filtered_prompts.append(truncated_text)
    batched_prompts = [
        filtered_prompts[i:i+args.bs] for i in range(0, len(filtered_prompts), args.bs)
    ]
    return batched_prompts 
    
def get_inference_engine(args: Namespace, model: MegatronModule) -> StaticInferenceEngine:
    """Utility to get the relevant backend for running inference

    This function will automatically choose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. TRT LLM Backend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model .

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_requests=args.inference_max_requests,
        inference_max_seq_length=args.inference_max_seq_length,
        # nccl_all_reduce_for_prefill=args.nccl_all_reduce_for_prefill
    )

    inference_context = StaticInferenceContext.from_config(inference_wrapper_config)

    inference_wrapped_model = GPTInferenceWrapper(
        model,
        inference_wrapper_config,
        inference_context
    )
    text_generation_controller = TextGenerationController(inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer)
    return StaticInferenceEngine(text_generation_controller=text_generation_controller)


async def generate(
    inference_engine: StaticInferenceEngine,
    sampling_params: SamplingParams,
    prompts: List[str],
) -> List[InferenceRequest]:
    async def collect_stream(prompt, request_id, stream_generator):
        print(f"Request {request_id}: {prompt}", end="", flush=True)
        prev_idx = 0
        async for output in stream_generator:
            print(output.generated_text[prev_idx:], end="", flush=True)
            prev_idx = len(output.generated_text)
        print()

    request_ids: List[str] = [
        inference_engine.add_request(
            prompt=prompt, sampling_params=sampling_params, streaming=True
        )
        for prompt in prompts
    ]
    stream_generators = [inference_engine.get_stream_generator(request_id) for request_id in request_ids]

    tasks = [
        asyncio.create_task(collect_stream(prompt, request_id, stream_generator))
        for (prompt, request_id, stream_generator) in zip(prompts, request_ids, stream_generators)
    ]

    await inference_engine.run_engine_async()
    await asyncio.gather(*tasks)

    results: List[InferenceRequest] = [
        inference_engine.scheduler.completed_request_pool[request_id] for request_id in request_ids
    ]

    return results

def main():
    """Main program."""

    # Note: The default args passed here can be overwritten by using appropriate params (check arguments.py file)
    # Micro batch size is not needed to be set by user. (It is calculated based on inference-batch-times-seqlen-threshold argument)
    initialize_megatron(
        extra_args_provider=add_static_inference_args, 
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'exit_on_missing_checkpoint': True,
        },
    )
    args = get_args()
    args.padded_vocab_size = args.vocab_size
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=False)
   

    ep_rank = mpu.get_expert_model_parallel_rank()
    model_dict = model[0].module.state_dict()
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
    print("keys",state.keys() )
    print("save empty weight")
    print("save to ",path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    torch.distributed.barrier()
    if ep_rank == 0:
        print("All ranks have finished saving.")
        
    # create file latest_checkpointed_iteration.txt
    tracker_file = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
    with open(tracker_file, "w") as f:
        f.write("1")
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()