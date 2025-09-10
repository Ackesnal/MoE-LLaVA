#!/bin/bash

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01

# RePaMoE specific arguments
FINETUNE_REPA_MODE=true
GATED_RATIO=0.5

JSON_FOLDER="/mnt/data/llava_data/train_json"
IMAGE_FOLDER="/mnt/data/llava_data/train_image"
cd ~/MoE-LLaVA
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun \
    --nproc_per_node=1 --nnodes=1 \
     moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.w1 mlp.w2 mlp.c_proj wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/MoE-LLaVA-Qwen-1.8B-4e \
    --version qwen \
    --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/MoE-LLaVA-Qwen-1.8B-4e-RePa-Save \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 20 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir" \
    --report_to wandb \
    --finetune_repa_mode $FINETUNE_REPA_MODE \
    --gated_ratio $GATED_RATIO 
