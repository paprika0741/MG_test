export MOE_TIME=0
export IDEAL=0
export DEBUG=0
export EPLB=0
export REPLICATE=0
export Online_Predict=0
export Async_Predict=1
export Step=1
export Layer_Time=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export Predictor_Path="/home/download/models/mg_core/Mixtral-8x7B-v0.1/router"

if [ "$Async_Predict" -eq 1 ]; then
    LOG_FILE="moe_infer_Async_Predict.log"
elif [ "$Online_Predict" -eq 1 ]; then
    LOG_FILE="moe_infer_Online_Predict.log"
else
    LOG_FILE="moe_infer.log"
fi
echo "Using log file: $LOG_FILE"

DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"
CHECKPOINT="/home/download/models/mg_core/Mixtral-8x7B-v0.1/mcore-TP1PP1EP4"
TOKENIZER_MODEL=/home/download/models/Mixtral-8x7B-v0.1/tokenizer.model
python  /home/CodeSpace/Megatron-LM-core_v0.12.0/utils/get_gate_weight.py --root_dir $CHECKPOINT --save_dir $Predictor_Path
 
export CUDA_DEVICE_MAX_CONNECTIONS=1
CUDA_VISIBLE_DEVICES=0,1,2,4  torchrun $DISTRIBUTED_ARGS ../../tools/run_text_generation_server.py   \
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
       --inference-max-seq-length 32768 \
       --transformer-impl transformer_engine  \
       --vocab-size 32000 \
       --moe-router-pre-softmax  \
       --moe-router-dtype "fp32"  2>&1 | tee $LOG_FILE

