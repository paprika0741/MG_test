# copy from  examples/inference/gpt/gpt_dynamic_inference_357m.sh
export MOE_TIME=0
export IDEAL=0
export DEBUG=1
MEGATRON_PATH=/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost"

CHECKPOINT="/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/mixtral-mcore-TP1PP1EP4Layer1"
TOKENIZER_MODEL="/home/ec2-user/CodeSpace/download/models/Mixtral-8x7B-v0.1/tokenizer.model"
start_time=$(date +%s) 
export CUDA_DEVICE_MAX_CONNECTIONS=1
export EXPERT_PATH=${CHECKPOINT}

: ${NUM_TOKENS_TO_PROMPT="8 32"}
: ${NUM_TOKENS_TO_GENERATE=256}
: ${INCOMING_REQUESTS_DURATION=10.}
: ${INCOMING_REQUESTS_PER_SEC=100.}

: ${INFERENCE_DYNAMIC_BATCHING_BUFFER_SIZE_GB=50.}
: ${INFERENCE_DYNAMIC_BATCHING_BUFFER_OVERFLOW_FACTOR=1.}
: ${INFERENCE_DYNAMIC_BATCHING_BUFFER_GUARANTEED_FRACTION=0.05}

: ${ENGINE=dynamic}

if [[ -v PROMPTS ]]; then
    ARGS+=" --prompts ${PROMPTS}"
else
    ARGS+=" \
        --num-tokens-to-prompt ${NUM_TOKENS_TO_PROMPT} \
        --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
        --incoming-requests-duration ${INCOMING_REQUESTS_DURATION} \
        --incoming-requests-per-sec ${INCOMING_REQUESTS_PER_SEC} \
    "
fi

torchrun $DISTRIBUTED_ARGS -m mixtral.dynamic_inferecne.gpt_dynamic_inference \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --expert-model-parallel-size 4 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model $TOKENIZER_MODEL \
       --use-mcore-models \
       --max-position-embeddings 32768 \
       --num-layers 1 \
       --load ${CHECKPOINT} \
       --hidden-size 4096 \
       --ffn-hidden-size 14336 \
       --num-attention-heads 32 \
       --normalization RMSNorm \
       --disable-bias-linear \
       --position-embedding-type rope \
       --no-position-embedding \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --group-query-attention \
       --num-query-groups 8 \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --seed 42 \
       --num-experts 8 \
       --moe-router-topk 2 \
       --moe-token-dispatcher-type alltoall \
       --moe-grouped-gemm \
       --rotary-base 1000000 \
       --no-rope-fusion \
       --no-gradient-accumulation-fusion \
       --inference-dynamic-batching \
       --inference-dynamic-batching-buffer-size-gb ${INFERENCE_DYNAMIC_BATCHING_BUFFER_SIZE_GB} \
       --inference-dynamic-batching-buffer-overflow-factor ${INFERENCE_DYNAMIC_BATCHING_BUFFER_OVERFLOW_FACTOR} \
       --inference-dynamic-batching-buffer-guaranteed-fraction ${INFERENCE_DYNAMIC_BATCHING_BUFFER_GUARANTEED_FRACTION} \
       --inference-max-seq-length 32768  