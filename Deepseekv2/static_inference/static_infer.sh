export MOE_TIME=0
export IDEAL=0
export DEBUG=0
export EPLB=0
export REPLICATE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

MEGATRON_PATH=/home/CodeSpace/Megatron-LM-core_v0.12.0
TOKENIZER_MODEL="/home/download/models/DeepSeek-V2-Lite"
CHECKPOINT="/home/download/models/mg_core/DeepSeek-V2-Lite/mcore-TP1PP1EP4"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export SAVE_TRACE_DIR="/home/download/models/mg_core/DeepSeek-V2-Lite/100_40/"

DISTRIBUTED_ARGS="--nproc_per_node 4  --nnodes 1"

MODEL_ARGS=" \
    --save-interval 100000 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --num-layers 27 \
    --hidden-size 2048 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 16 \
    --kv-channels 16 \
    --multi-latent-attention \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-head-dim 128 \
    --qk-layernorm \
    --qk-pos-emb-head-dim 64 \
    --num-experts 64 \
    --moe-layer-freq ([0]+[1]*26) \
    --moe-ffn-hidden-size 1408 \
    --moe-grouped-gemm \
    --moe-router-score-function softmax \
    --moe-router-topk 6 \
    --moe-router-topk-scaling-factor 1.0 \
    --moe-router-pre-softmax \
    --moe-shared-expert-intermediate-size 2816 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-token-drop-policy probs \
    --moe-router-load-balancing-type seq_aux_loss \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 3200 \
    --attention-softmax-in-fp32 \
    --use-mcore-models \
    --use-rotary-position-embeddings \
    --rotary-percent 1.0 \
    --rotary-base 10000 \
    --rotary-scaling-factor 40 \
    --mscale 0.707 \
    --mscale-all-dim 0.707 \
    --sequence-parallel \
    --group-query-attention \
    --num-query-groups 16 \
    --moe-router-dtype "fp64"
"

torchrun $DISTRIBUTED_ARGS /home/CodeSpace/Megatron-LM-core_v0.12.0/mixtral/static_inference/gpt_static_inference.py  \
       --no-masked-softmax-fusion \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --expert-model-parallel-size 4 \
       --tokenizer-model $TOKENIZER_MODEL \
       --vocab-size 102400 \
       --no-rope-fusion \
       --hf_path $TOKENIZER_MODEL \
       --no-gradient-accumulation-fusion \
       --load ${CHECKPOINT} \
       $MODEL_ARGS