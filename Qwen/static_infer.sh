export MOE_TIME=0
export IDEAL=0
export DEBUG=0
export EPLB=0
export REPLICATE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

MEGATRON_PATH=/home/CodeSpace/Megatron-LM-core_v0.12.0
TOKENIZER_MODEL="/home/download/models/Qwen1.5-MoE-A2.7B-Chat"
CHECKPOINT="/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat/mcore-TP1PP1EP4"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export SAVE_TRACE_DIR="/home/download/models/mg_core/DeepSeek-V2-Lite/100_40/"

DISTRIBUTED_ARGS="--nproc_per_node 4  --nnodes 1"

MODEL_ARGS=" \
       --tokenizer-type HuggingFaceTokenizer  \
       --tokenizer-model $TOKENIZER_MODEL \
       --use-mcore-models \
       --max-position-embeddings 32768 \
       --num-layers 24 \
       --hidden-size 2048 \
       --moe-ffn-hidden-size 1408 \
       --moe-shared-expert-intermediate-size 5632 \
       --num-attention-heads 16 \
       --normalization RMSNorm \
        --norm-epsilon  1e-6 \
       --disable-bias-linear \
       --position-embedding-type rope \
       --no-position-embedding \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --seed 42 \
       --num-experts 60 \
       --moe-router-topk  4 \
       --group-query-attention \
       --num-query-groups 16 \
       --moe-token-dispatcher-type alltoall \
       --moe-grouped-gemm \
       --rotary-base 1000000 \
       --no-rope-fusion \
       --no-gradient-accumulation-fusion \
       --max-batch-size 8 \
       --inference-max-seq-length 4096 \
       --transformer-impl transformer_engine  \
       --vocab-size 151936 \
       --load ${CHECKPOINT} \
       --moe-use-shared-expert-gate \
       --add-qkv-bias  \
       --moe-router-pre-softmax   \
       --moe-router-dtype "fp32" 
"

torchrun $DISTRIBUTED_ARGS /home/CodeSpace/Megatron-LM-core_v0.12.0/mixtral/static_inference/gpt_static_inference.py  \
       --no-masked-softmax-fusion \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --expert-model-parallel-size 4 \
       --hf_path $TOKENIZER_MODEL \
       $MODEL_ARGS