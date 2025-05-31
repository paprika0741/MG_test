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
from megatron.core.transformer.module import MegatronModule

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

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

    return parser


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

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=False)
    model = model[0]

    args = get_args()

    inference_engine = get_inference_engine(args, model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        top_n_logprobs=args.top_n_logprobs,
    )

    requests = build_requests(args, get_tokenizer())
    prompts = [ r.prompt_text for r in requests ]

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        inference_engine.generate(
                prompts=prompts, sampling_params=sampling_params
            )
    start_time = time.perf_counter()
    if args.stream:
        results: List[InferenceRequest] = asyncio.run(generate(inference_engine, sampling_params, prompts))
    else:
        results: List[InferenceRequest] = inference_engine.generate(
            prompts=prompts, sampling_params=sampling_params,
        )
    end_time = time.perf_counter()
    latency = end_time - start_time

    if torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
            result_dict = {
                'id': result.request_id,
                'input_prompt': result.prompt,
                'generated_text': result.generated_text,
                'generated_tokens': result.generated_tokens,
                'latency': latency,
            }
            if sampling_params.top_n_logprobs > 0 :
                result_dict['generated_top_n_logprobs'] = result.generated_top_n_logprobs
            if sampling_params.return_log_probs:
                response_logprobs = result.prompt_log_probs + result.generated_log_probs
                result_dict["logprobs"] = response_logprobs

        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            # Tensors cannot be serialized so we move these to CPU
            result_dict['generated_tokens'] = result_dict['generated_tokens'].cpu().numpy().tolist()
            results_as_json = json.dumps(result_dict)
            output_dir = os.path.dirname(args.output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(args.output_path, 'w') as f:
                json.dump(results_as_json, f)

    # Print unique prompts + outputs.
    if torch.distributed.get_rank() == 0:

        print("~~~~ Unique prompts + outputs. ~~~~")

        # Map results by their prompt.
        from collections import defaultdict
        unique_prompt_map = defaultdict(list)
        for result_idx, result in enumerate(results):
            unique_prompt_map[result.prompt].append(result_idx)

        # Print unique prompts + outputs.
        for unique_idx, (prompt_text, result_idxs) in enumerate(unique_prompt_map.items()):
            result_idx = result_idxs[0]
            result = results[result_idx]
            print(f"{unique_idx}/{len(unique_prompt_map)} [{len(result_idxs)}]. {prompt_text} ... %s" % result.generated_text.replace("\n", "\\n"))


    stats = torch.cuda.memory_stats()
    print("static | cg %d | %s | reqs %d [ batch %d ] ... mem %.1f/%.1f ... time %.3f." % (
        args.enable_cuda_graph,
        (
            f"<user prompts>"
            if args.prompts else
            "<auto prompts> %s, %d, %.1e, %.1e" % (
                "(%s)" % " ".join(map(str, args.num_tokens_to_prompt)),
                args.num_tokens_to_generate,
                args.incoming_requests_duration,
                args.incoming_requests_per_sec,
            )
        ),
        len(requests),
        args.inference_max_requests,
        stats["allocated_bytes.all.peak"] / (1024**3),
        stats["reserved_bytes.all.peak"] / (1024**3),
        latency,
    ))

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()