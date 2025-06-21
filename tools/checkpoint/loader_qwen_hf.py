# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import transformers
from tqdm import tqdm
import types


def add_arguments(parser):
    group = parser.add_argument_group(title='Mixtral HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def load_args_from_checkpoint(args):
    # [modified] 
    # support other models
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.load)

    # Update Megatron args.
    args.untie_embeddings_and_output_weights = True
    args.seq_length = 4096
    args.global_batch_size = 1024
    args.iteration = 1 # '0', 'release' don't work
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    args.bf16 = True
    args.add_bias_linear = False
    args.normalization = "RMSNorm"
    args.tokenizer_type = "Llama2Tokenizer"
    args.disable_bias_linear = True

    args.max_position_embeddings = config.max_position_embeddings
    args.hidden_size = config.hidden_size
    args.num_attention_heads = config.num_attention_heads
    args.num_layers = config.num_hidden_layers
    args.norm_epsilon = config.rms_norm_eps
    args.vocab_size = config.vocab_size
    args.padded_vocab_size = config.vocab_size
    args.mixtral = config
 
    args.moe_ffn_hidden_size = config.moe_intermediate_size
    args.moe_shared_expert_intermediate_size = config.shared_expert_intermediate_size
    args.moe_router_dtype = "fp32" 
    #TODO:
    args.moe_router_pre_softmax = True 
    args.moe_use_shared_expert_gate = True  # [modified]
    args.ffn_hidden_size = config.intermediate_size
    args.num_experts = config.num_experts
    args.sequence_parallel = True
    args.num_experts_per_tok = config.num_experts_per_tok
    args.add_qkv_bias = True  # [modified]
    args.apply_rope_fusion = False
    args.gradient_accumulation_fusion = False
    if config.num_key_value_heads:
        args.group_query_attention = True
        args.num_query_groups = config.num_key_value_heads

def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major >= 4 and minor >= 36

def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)

def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    model.output_layer.weight.data.copy_(hf_model.lm_head.weight)

def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    num_heads = args.num_attention_heads // tp
    num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads) // tp
    num_querys_per_group = num_heads // num_query_groups
    dim = args.kv_channels
    assert num_heads % num_querys_per_group == 0
    print("num_query_groups",num_query_groups)
    print("dim",dim)
    print("num_querys_per_group",num_querys_per_group)
    print("num_querys_per_group*dim", num_querys_per_group*dim)
    # Copy weights (re-order dimensions for Megatron).
    attn.linear_qkv.weight.data.copy_(torch.cat([
        hf_attn.q_proj.weight.reshape((num_query_groups, num_querys_per_group*dim, -1)),
        hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
        hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)

def set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''
    # NOTE hf_layer weigt keys are different between different models
    layer.mlp.router.weight.data.copy_(hf_layer.mlp.gate.weight)

    mcore_experts = layer.mlp.experts.local_experts
    hf_experts = hf_layer.mlp.experts
    for expert_idx in range(args.num_experts):
        mcore_experts[expert_idx].linear_fc1.weight.data.copy_(
            torch.cat([
                hf_experts[expert_idx].gate_proj.weight,
                hf_experts[expert_idx].up_proj.weight
            ], dim=0)
        )
        mcore_experts[expert_idx].linear_fc2.weight.data.copy_(
            hf_experts[expert_idx].down_proj.weight
        )
    # shared experts
    layer.mlp.shared_experts.linear_fc1.weight.data.copy_(
        torch.cat([
                hf_layer.mlp.shared_expert.gate_proj.weight,
                hf_layer.mlp.shared_expert.up_proj.weight
            ], dim=0)
    )
    layer.mlp.shared_experts.linear_fc2.weight.data.copy_(
        hf_layer.mlp.shared_expert.down_proj.weight
    )
    print("Shared  experts")
    for name, param in layer.mlp.shared_experts.named_parameters():
        print(f"name {name}: {param.shape}")
    if hasattr(layer.mlp.shared_experts, "gate_weight") and layer.mlp.shared_experts.gate_weight is not None:
        hf_weight = hf_layer.mlp.shared_expert_gate.weight
        assert not torch.isnan(hf_weight).any(), "NaN detected in hf_layer.mlp.shared_expert_gate.weight"
        assert not torch.isinf(hf_weight).any(), "Inf detected in hf_layer.mlp.shared_expert_gate.weight"
        # Copy safely
        # layer.mlp.shared_experts.gate_weight is a tensor
        with torch.no_grad():
            layer.mlp.shared_experts.gate_weight.copy_(hf_weight)
            print("Check gate_weight after copy:")
        assert not torch.isnan(layer.mlp.shared_experts.gate_weight).any(), "NaN in gate_weight"
        assert not torch.isinf(layer.mlp.shared_experts.gate_weight).any(), "Inf in gate_weight"

def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)

    layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)

def load_checkpoint_to_model(args):
    '''Set model params.'''

    from pretrain_gpt import model_provider
    from transformers import AutoModelForCausalLM, AutoConfig

    # Load Huggingface model.
    print("[DEBUG] Loading HF model from:", args.load)

    hf_model = AutoModelForCausalLM.from_pretrained(args.load, device_map="cpu")
    print("hf_model",hf_model)
    print("[DEBUG] HF model loaded.")

    # Init Megatron model.
    print("[DEBUG] Initializing Megatron model...")

    model = model_provider(True, True).to(args.params_dtype)
    print("model",model)
    print("[DEBUG] Megatron model initialized.")

    # Set model state.
    print("[DEBUG] Setting embedding weights...")
    print("[DEBUG] set_preprocess_state")

    set_preprocess_state(args, model, hf_model)
    
    print("[DEBUG] set_postprocess_state")

    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        print(f"[DEBUG] setting layer {layer_idx}")
        set_layer_state(args, model, hf_model, layer_idx)
    print("[DEBUG] All transformer layers set.")
    return model


def _load_checkpoint(queue, args):

    # Llama-2 requires HF transformers >=4.31.0.
    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--use-mcore-models',
                '--disable-bias-linear',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--mock-data', # To pass the "blend data checks" in arguments.py
                '--transformer-impl', 'transformer_engine',
                '--load', args.load_dir,
                '--no-one-logger',
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('disable_bias_linear')
    check_for_arg('params_dtype')
    check_for_arg('swiglu')

    # Determine how to make our models.
    assert args.model_type == 'GPT', 'Llama-2 is a GPT model.'
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    fused_kernels.load(margs)

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = margs.vocab_size # skips padding in saver
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.num_experts = margs.num_experts
    # [modified]
    md.qkv_bias =  True
    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.embedding, 'position_embeddings')

    queue_put("embeddings", message)

    for layer_idx in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.decoder.layers[layer_idx]
        message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
        message["post norm weight"] = layer.pre_mlp_layernorm.weight.data

        # Simple concat of the rest.
        message["qkv weight"] = layer.self_attention.linear_qkv.weight.data
        # [modified]
        message["qkv bias"] = layer.self_attention.linear_qkv.bias.data
        message["dense weight"] = layer.self_attention.linear_proj.weight.data

        # Grab all parallel tensors for this layer.
        layer = model.decoder.layers[layer_idx]
        experts = layer.mlp.experts.local_experts

        message["router weight"] = layer.mlp.router.weight.data
        if md.swiglu:
            chunked_mlp_l0_weight =  [torch.chunk(local_expert.linear_fc1.weight.data, 2, dim=0) for local_expert in experts]
            message["mlp l0 weight W"] = torch.stack([local_weight[0] for local_weight in chunked_mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.stack([local_weight[1] for local_weight in chunked_mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.stack([local_expert.linear_fc1.weight.data for local_expert in experts])
        message["mlp l1 weight"] = torch.stack([local_expert.linear_fc2.weight.data for local_expert in experts], dim=0)
        # add shared experts
        # message["mlp shared experts fc1 weight"] = layer.mlp.shared_experts.linear_fc1.weight.data
        # message["mlp shared experts fc2 weight"] = layer.mlp.shared_experts.linear_fc2.weight.data
        # message["mlp shared experts gate weight"] = layer.mlp.shared_experts.gate_weight
        # print("[Debug] Keys sent from loader:", list(message.keys()))

        
        queue_put(f"transformer layer {layer_idx}", message)

    queue_put("final norm", {
        "weight": model.decoder.final_layernorm.weight.data,
    })

    if md.output_layer:
        queue_put("output layer", {
            "weight": model.output_layer.weight.data
        })

    queue.put("done")

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
