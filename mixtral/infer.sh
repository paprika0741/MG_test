export MOE_TIME=1
export IDEAL=0
export DEBUG=1
export EPLB=0
export REPLICATE=0
EP=4
DISTRIBUTED_ARGS="--nproc_per_node ${EP} \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"
CHECKPOINT="/home/download/models/mg_core/Mixtral-8x7B-v0.1/mcore-TP1PP1EP${EP}"
# CHECKPOINT=/mnt/data/mixtral-mcore-TP1PP1EP4Layer8
TOKENIZER_MODEL=/home/download/models/Mixtral-8x7B-v0.1/tokenizer.model
export EXPERT_PATH=${CHECKPOINT}
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun $DISTRIBUTED_ARGS ../tools/run_text_generation_server.py   \
       --port 5000 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --expert-model-parallel-size ${EP} \
       --load ${CHECKPOINT}  \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model $TOKENIZER_MODEL \
       --use-mcore-models \
       --max-position-embeddings 32768 \
       --num-layers 8 \
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
       --max-batch-size 8 \
       --inference-max-seq-length 32768 \
       --transformer-impl transformer_engine  \
       --vocab-size 32000 \
       --moe-router-pre-softmax  \
       --moe-router-dtype "fp32" 

