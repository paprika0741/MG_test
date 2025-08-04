export MOE_TIME=0
export IDEAL=0
export DEBUG=0
export EPLB=0
export REPLICATE=0
 
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0"
MEGATRON_PATH=/home/ubuntu/CodeSpace/MG_test

export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

CHECKPOINT="/opt/dlami/nvme/weight/mcore-TP1PP1EP8_new"
TOKENIZER_MODEL=/home/ubuntu/CodeSpace/MG_test/checkpoints/mistralai/Mixtral-8x7B-v0.1/tokenizer.model
export EXPERT_PATH=${CHECKPOINT}
export CUDA_DEVICE_MAX_CONNECTIONS=1
HF_PATH=/home/ubuntu/CodeSpace/MG_test/checkpoints/mistralai/Mixtral-8x7B-v0.1/


num=64
bs=2
inlen=32
outlen=16
MY_INFER_ARGS="--num ${num} --bs ${bs} --input_len ${inlen} --output_len ${outlen}"
export SAVE_TRACE_DIR="/home/ubuntu/CodeSpace/MG_test/experiment/Mixtral-8x7B-v0.1/bs${bs}_in${inlen}_out${outlen}_num${num}_new/"
echo "[INFO] Save trace path: $SAVE_TRACE_DIR"

torchrun $DISTRIBUTED_ARGS ../gpt_static_inference.py   $MY_INFER_ARGS \
      --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --expert-model-parallel-size 8 \
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
       --max-batch-size 8 \
       --inference-max-seq-length 256 \
       --transformer-impl transformer_engine  \
       --vocab-size 32000 \
       --moe-router-pre-softmax  \
       --moe-router-dtype "fp32" \
       --prompts "Notably, the degree of acceleration is influenced by the numerical relationship between the number of experts and GPUs in Expert Parallelism. In Mixtral-8×7B-Instruct, one-to-one deployment maximizes the effectiveness of capacity-aware inference. To use a different security group, choose Select existing security group and choose an existing security group."   \
       --num-tokens-to-generate 1 \
       --hf_path ${HF_PATH} \
       --load ${CHECKPOINT}
 
echo "[INFO] Save trace path: $SAVE_TRACE_DIR"
