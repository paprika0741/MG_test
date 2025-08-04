export MOE_TIME=0
export IDEAL=0
export DEBUG=1
export EPLB=0
export REPLICATE=0
 
MEGATRON_PATH=/home/ubuntu/CodeSpace/MG_test
TOKENIZER_MODEL=/home/ubuntu/CodeSpace/MG_test/checkpoints/mistralai/Mixtral-8x7B-v0.1/tokenizer.model
HF_FORMAT_DIR=/home/ubuntu/CodeSpace/MG_test/checkpoints/mistralai/Mixtral-8x7B-v0.1
MEGATRON_FORMAT_DIR="/opt/dlami/nvme/weight/mcore-TP1PP1EP8_new"

export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
 
 

DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost"

MODEL_ARGS=" \
       --save-interval 100000 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model $TOKENIZER_MODEL \
       --use-mcore-models \
       --max-position-embeddings 32768 \
       --num-layers 32 \
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
       --inference-max-seq-length 256 \
       --transformer-impl transformer_engine  \
       --vocab-size 32000   \
       --moe-router-pre-softmax  \
       --moe-router-dtype "fp32" 
"

torchrun $DISTRIBUTED_ARGS  create_model.py  \
        --no-masked-softmax-fusion \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --expert-model-parallel-size 8 \
       --no-rope-fusion \
       --no-gradient-accumulation-fusion \
       --save $MEGATRON_FORMAT_DIR \
       $MODEL_ARGS

if [ $? -eq 0 ]; then
    echo "Modify weights: $MEGATRON_FORMAT_DIR"
    python modify_dict.py --root_dir $MEGATRON_FORMAT_DIR --hf_path $HF_FORMAT_DIR
    MODIFY_STATUS=$?

#     if [ $MODIFY_STATUS -eq 0 ]; then
#         echo "Check weights: $MEGATRON_FORMAT_DIR"
#         python check_weight_hf.py --root_dir $MEGATRON_FORMAT_DIR --hf_path $HF_FORMAT_DIR
#     else
#         echo "modify_dict.py failed, skipping check_weight_hf.py."
#     fi
# else
#     echo "Conversion failed, skipping modify_dict.py and check_weight_hf.py."
# fi

