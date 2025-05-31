export MOE_TIME=1
export IDEAL=0
export DEBUG=1
export EPLB=0
export REPLICATE=0
export OLMoE=1

DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1"
         
CHECKPOINT="/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/OLMoE/mcore-TP1PP1EP4Layer1"
TOKENIZER_MODEL="/home/ec2-user/models/OLMoE-1B-7B-0125-Instruct" 

export CUDA_DEVICE_MAX_CONNECTIONS=1

 
torchrun $DISTRIBUTED_ARGS ./gpt_static_inference.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --expert-model-parallel-size 4 \
       --load ${CHECKPOINT}  \
       --tokenizer-type HuggingFaceTokenizer  \
       --tokenizer-model $TOKENIZER_MODEL \
       --use-mcore-models \
       --max-position-embeddings 4096 \
       --num-layers 1 \
       --hidden-size 2048 \
       --ffn-hidden-size 1024 \
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
       --num-experts 64 \
       --moe-router-topk 8 \
       --group-query-attention \
       --num-query-groups 16 \
       --moe-token-dispatcher-type alltoall \
       --moe-grouped-gemm \
       --rotary-base 10000 \
       --no-rope-fusion \
       --no-gradient-accumulation-fusion \
       --max-batch-size 8 \
       --inference-max-seq-length 4096 \
       --transformer-impl transformer_engine   \
       --vocab-size 50304   \
       --qk-layernorm \
       --prompts "Notably, the degree of acceleration is influenced by the numerical relationship between the number of experts and GPUs in Expert Parallelism. In Mixtral-8×7B-Instruct, one-to-one deployment maximizes the effectiveness of capacity-aware inference. To use a different security group, choose Select existing security group and choose an existing security group."   \
       --num-tokens-to-generate 1 \
       
 

