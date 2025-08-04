export DEBUG=0
export MOE_TIME=0
export IDEAL=0

DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1"
MEGATRON_PATH=/home/CodeSpace/Megatron-LM-core_v0.12.0
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
CHECKPOINT="/home/download/models/mg_core/Mixtral-8x7B-v0.1/mcore-TP1PP1EP4"
TOKENIZER_MODEL=/home/download/models/Mixtral-8x7B-v0.1/tokenizer.model
#   --load ${CHECKPOINT}  \
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun $DISTRIBUTED_ARGS ./gpt_static_inference.py    \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --expert-model-parallel-size 4 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model $TOKENIZER_MODEL \
       --use-mcore-models \
       --max-position-embeddings 32768 \
       --num-layers 2 \
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
       --vocab-size 32000 \
       --prompts "Notably, the degree of acceleration is influenced by the numerical relationship between the number of experts and GPUs in Expert Parallelism. In Mixtral-8×7B-Instruct, one-to-one deployment maximizes the effectiveness of capacity-aware inference. To use a different security group, choose Select existing security group and choose an existing security group."   \
       --num-tokens-to-generate 1 \
       --hf_path "/home/download/models/Mixtral-8x7B-v0.1/" \
       --moe-router-pre-softmax  \
       --moe-router-dtype "fp32" 
       
 

