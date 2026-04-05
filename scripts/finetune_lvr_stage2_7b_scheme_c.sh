#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# STAGE1_STEPS=2500
# CHKPT_PATH="stage1_checkpoints/Qwen2.5-VL-7B-Instruct/Stage1_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128/checkpoint-${STAGE1_STEPS}/"
CHKPT_PATH="/root/autodl-fs/LVR-7B"
# data configs
DATA_PATH="/root/autodl-fs/data/lvr_data/virl39k.json"
IMAGE_FOLDER="/root/autodl-fs/data/"
OUTPUT_DIR="stage2_checkpoints_39k_scheme_c/"

# training configs
FREEZE_VISION=True
FREEZE_MERGER=True
DECODING_STRATEGY="steps"
LVR_STEPS=8
LR=5e-6
TEMP=0.9

# RLVR objective scheme C configs
BETA=0.02
SIGMA=7.2
USE_BETA_HAT=False
T_MAX=1024

# model configs
export WANDB_PROJECT="LVR-Qwen25-VL-7B-RLVR-STAGE-2-SCHEME-C-39k"
export WANDB_MODE="offline"
RUN_NAME="Stage2_7B_schemeC_decodingBy${DECODING_STRATEGY}_max${LVR_STEPS}lvrSteps_LR${LR}_TEMP${TEMP}_beta${BETA}_sigma${SIGMA}_tmax${T_MAX}_stage1Steps${STAGE1_STEPS}"

deepspeed src/train/train_grpo.py \
    --run_name "$RUN_NAME" \
    --deepspeed scripts/zero2.json \
    --online_checkpoint True \
    --checkpoint_name $CHKPT_PATH \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --freeze_vision_tower $FREEZE_VISION \
    --freeze_merger $FREEZE_MERGER \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --beta $BETA \
    --sigma $SIGMA \
    --use_beta_hat $USE_BETA_HAT \
    --t_max $T_MAX \
    --temperature $TEMP \
    --num_train_epochs 2 \
    --num_generations 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_completion_length 512 \
    --max_prompt_length 4096 \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((2560 * 28 * 28)) \
    --learning_rate $LR \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 50 \
    --dataloader_num_workers 8 \
    --decoding_strategy $DECODING_STRATEGY \
    --lvr_steps $LVR_STEPS
