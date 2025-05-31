export DEBUG=0
TOKENIZER_MODEL=/mnt/data/Qwen1.5-MoE-A2.7B-Chat
MEGATRON_PATH=/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE="1"
TARGET_EP_SIZE="4"
TARGET_PP_SIZE="1"
 
HF_FORMAT_DIR=/mnt/data/Qwen1.5-MoE-A2.7B-Chat
MEGATRON_FORMAT_DIR=/mnt/data/mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}
# MEGATRON_FORMAT_DIR=/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/Qwen/mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}Layer1


python ../tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_qwen_hf  \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL} \
--saver-transformer-impl transformer_engine \
 
 
if [ $? -eq 0 ]; then
    echo "Modify key of weights: $MEGATRON_FORMAT_DIR"
    python modify_dict.py --root_dir $MEGATRON_FORMAT_DIR
else
    echo "Conversion failed, skipping modify_dict.py."
fi