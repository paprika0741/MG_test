export MOE_TIME=1
export IDEAL=0
export DEBUG=1
export EPLB=0
export REPLICATE=0
DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"
# CHECKPOINT="/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/Qwen/mcore-TP1PP1EP4Layer1"
CHECKPOINT=/mnt/data/mcore-TP1PP1EP4/
TOKENIZER_MODEL=/mnt/data/Qwen1.5-MoE-A2.7B-Chat

export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun $DISTRIBUTED_ARGS ../tools/run_text_generation_server.py   \
       --port 5000 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --expert-model-parallel-size 4 \
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
       --add-qkv-bias  
 
 
