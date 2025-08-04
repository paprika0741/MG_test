export MOE_TIME=0
export IDEAL=0
export DEBUG=1
export EPLB=0
export REPLICATE=0
MEGATRON_PATH=/home/CodeSpace/Megatron-LM-core_v0.12.0
TOKENIZER_MODEL="/home/download/models/Qwen1.5-MoE-A2.7B-Chat"
HF_FORMAT_DIR="/home/download/models/Qwen1.5-MoE-A2.7B-Chat"
MEGATRON_FORMAT_DIR="/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat_test/mcore-TP1PP1EP4"

export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

rm -rf $MEGATRON_FORMAT_DIR

DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost"

 

torchrun $DISTRIBUTED_ARGS  create_model.py  \
       --save-interval 100000 \
        --no-masked-softmax-fusion \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --expert-model-parallel-size 4 \
       --tokenizer-model $TOKENIZER_MODEL \
       --vocab-size 102400 \
       --no-rope-fusion \
       --no-gradient-accumulation-fusion \
       --save $MEGATRON_FORMAT_DIR \
        --tokenizer-type HuggingFaceTokenizer  \
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
       --inference-max-seq-length 4096 \
       --transformer-impl transformer_engine  \
       --vocab-size 151936 \
        --moe-use-shared-expert-gate \
       --add-qkv-bias  



if [ $? -eq 0 ]; then
    echo "Modify key of weights: $MEGATRON_FORMAT_DIR"
    python modify_dict.py --root_dir $MEGATRON_FORMAT_DIR --hf_path $HF_FORMAT_DIR
    MODIFY_STATUS=$?

    if [ $MODIFY_STATUS -eq 0 ]; then
        echo "Check weights: $MEGATRON_FORMAT_DIR"
        python check_weight_hf.py --root_dir $MEGATRON_FORMAT_DIR --hf_path $HF_FORMAT_DIR
    else
        echo "modify_dict.py failed, skipping check_weight_hf.py."
    fi
else
    echo "Conversion failed, skipping modify_dict.py and check_weight_hf.py."
fi

