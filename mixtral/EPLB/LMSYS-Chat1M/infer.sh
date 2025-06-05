export DEBUG=1
export MOE_TIME=1
export IDEAL=1
export SKEW=1
export EPLB=0
export REPLICATE=0
LOG_FILE="moe_infer_ideal${IDEAL}_skew${SKEW}_eplb${EPLB}.log"
echo "DEBUG=$DEBUG"
echo "MOE_TIME=$MOE_TIME"
echo "IDEAL=$IDEAL"
echo "SKEW=$SKEW"
echo "EPLB=$EPLB"
echo "REPLICATE=$REPLICATE"
echo "LOG_FILE = $LOG_FILE"
 
if [ "$REPLICATE" -eq 1 ] && [ "$EPLB" -eq 1 ]; then
    echo "Error: REPLICATE and EPLB cannot both be 1."
    exit 1
fi

DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"
CHECKPOINT=/mnt/data/mixtral-mcore-TP1PP1EP4Layer8
TOKENIZER_MODEL=/home/ec2-user/CodeSpace/download/models/Mixtral-8x7B-v0.1/tokenizer.model
export EXPERT_PATH=${CHECKPOINT}
export CUDA_DEVICE_MAX_CONNECTIONS=1

start_time=$(date +%s) 
torchrun $DISTRIBUTED_ARGS ../../../tools/run_text_generation_server.py   \
       --port 5000 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --expert-model-parallel-size 4 \
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
       --vocab-size 32000 \
       --inference-max-seq-length 32768 2>&1 | tee $LOG_FILE
end_time=$(date +%s)  
elapsed=$((end_time - start_time))
echo "Total runtime: ${elapsed} seconds" | tee -a $LOG_FILE
