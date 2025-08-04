export MOE_TIME=0
export IDEAL=0
export DEBUG=0
export EPLB=0
export REPLICATE=0

DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1"
MEGATRON_PATH=/home/CodeSpace/Megatron-LM-core_v0.12.0
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
CHECKPOINT="/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat/mcore-TP1PP1EP4"
TOKENIZER_MODEL="/home/download/models/Qwen1.5-MoE-A2.7B-Chat"
export EXPERT_PATH=${CHECKPOINT}
export CUDA_DEVICE_MAX_CONNECTIONS=1
HF_PATH=/home/download/models/Qwen1.5-MoE-A2.7B-Chat


num=64
bs=3
inlen=32
outlen=16
MY_INFER_ARGS="--num ${num} --bs ${bs} --input_len ${inlen} --output_len ${outlen}"
export SAVE_TRACE_DIR="/home/CodeSpace/Megatron-LM-core_v0.12.0/fmoe_experiment/Qwen/bs${bs}_in${inlen}_out${outlen}_num${num}/"
echo "[INFO] Save trace path: $SAVE_TRACE_DIR"

torchrun $DISTRIBUTED_ARGS ../gpt_static_inference.py   $MY_INFER_ARGS \
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
       --add-qkv-bias   \
       --moe-router-pre-softmax  \
       --moe-router-dtype "fp32" \
       --prompts "Notably, the degree of acceleration is influenced by the numerical relationship between the number of experts and GPUs in Expert Parallelism. In Mixtral-8×7B-Instruct, one-to-one deployment maximizes the effectiveness of capacity-aware inference. To use a different security group, choose Select existing security group and choose an existing security group."   \
       --num-tokens-to-generate 1 \
       --hf_path ${HF_PATH} \
      --load ${CHECKPOINT}  
 
echo "[INFO] Save trace path: $SAVE_TRACE_DIR"
