export DEBUG=0
TOKENIZER_MODEL=/home/download/models/Mixtral-8x7B-v0.1/tokenizer.model
MEGATRON_PATH=/home/CodeSpace/Megatron-LM-core_v0.12.0
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE="1"
TARGET_EP_SIZE="4"
TARGET_PP_SIZE="1"
 
HF_FORMAT_DIR=/home/download/models/Mixtral-8x7B-v0.1/
MEGATRON_FORMAT_DIR="/home/download/models/mg_core/Mixtral-8x7B-v0.1/mcore-TP1PP1EP4"


python ../tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_mixtral_hf  \
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
    python modify_dict.py --root_dir $MEGATRON_FORMAT_DIR --hf_path $HF_FORMAT_DIR

    if [ $? -eq 0 ]; then
        echo "Check weights: $MEGATRON_FORMAT_DIR"
        python check_weight_hf.py --root_dir $MEGATRON_FORMAT_DIR --hf_path $HF_FORMAT_DIR
    else
        echo "modify_dict.py failed, skipping check_weight_hf.py."
    fi

else
    echo "Conversion failed, skipping modify_dict.py and check_weight_hf.py."
fi
