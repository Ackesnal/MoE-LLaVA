#!/bin/bash
#SBATCH --job-name=finetune_repa_moe
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=logs/log_finetune_repa_moe_v1.1_%j.out
#SBATCH --error=logs/log_finetune_repa_moe_v1.1_%j.err

# Load required modules

module load gcc/12.3.0
module load cuda/12.4.0
module load cudnn/9.3.0-cu12
module load miniconda3/23.5.2
module load ninja/1.11.1
module load sqlite/3.43.1
module load nccl/2.20.5-cu124
# module load openmpi/4.1.4-ofed54

# Change to working directory
cd /home/li309/pct_code/moe/MoE-LLaVA

# Activate conda environment
source activate /home/li309/pct_code/venv/moellava-test2

# Set MoE parameters
moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01

# RePaMoE specific arguments
FINETUNE_REPA_MODE=true
REPA_GATED_RATIO=1.0

# Set data paths
JSON_FOLDER="/scratch3/li309/data/llava_data/train_json"
IMAGE_FOLDER="/scratch3/li309/data/llava_data/train_data"

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH

# export NCCL_P2P_DISABLE=1
# export NCCL_P2P_LEVEL=PCI

function makehostfile() {
perl -e '
  my $step = $ENV{"SLURM_STEP_GPUS"} // "";
  my $slots = $ENV{"SLURM_GPUS_ON_NODE"} // 0;
  $slots ||= scalar(split(/,/, $step)) if $step ne "";
  $slots = 4 if !$slots;

  my @nodes = split /\n/, qx{scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}};
  print map { "$_ slots=$slots\n" } @nodes;
'
}

makehostfile > myhostfile

cat myhostfile

export MASTER_PORT=33789
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export WORLD_SIZE=$SLURM_NTASKS
# export RANK=$SLURM_PROCID
# export LOCAL_RANK=$SLURM_LOCALID
# export NODE_RANK=$SLURM_NODEID

echo "VERSION: 1.1"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"
#echo "WORLD_SIZE: $WORLD_SIZE"
#echo "RANK: $RANK"
#echo "GLOBAL_RANK: $GLOBAL_RANK"
#echo "LOCAL_RANK: $LOCAL_RANK"
#echo "NODE_RANK: $NODE_RANK"
#echo "SLURM_NNODES: $SLURM_NNODES"
#echo "SLURM_NTASKS: $SLURM_NTASKS"
#echo "SLURM_PROCID: $SLURM_PROCID"
#echo "SLURM_LOCALID: $SLURM_LOCALID"

# 仅在 node rank 0 上启动 deepspeed
HOSTLIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
THIS_HOST=$(hostname -s)
NODE_RANK_CALC=${SLURM_NODEID:-}
if [ -z "$NODE_RANK_CALC" ]; then
  NODE_RANK_CALC=0
  for i in "${!HOSTLIST[@]}"; do
    if [ "${HOSTLIST[$i]}" = "$THIS_HOST" ]; then NODE_RANK_CALC=$i; break; fi
  done
fi
export NODE_RANK=$NODE_RANK_CALC
echo "Computed NODE_RANK: $NODE_RANK"

if [ "$NODE_RANK" -eq 0 ]; then
  echo "Launching DeepSpeed on NODE_RANK 0 (host: $THIS_HOST)"

  # Run training
  WANDB_MODE=offline HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --launcher=slurm --hostfile=myhostfile \
    --num_gpus=$SLURM_GPUS_ON_NODE --num_nodes=$SLURM_NNODES \
    moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules gate_proj up_proj down_proj wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/MoE-LLaVA-StableLM-1.6B-4e \
    --version stablelm \
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
    --output_dir ./finetuned_checkpoints/MoE-LLaVA-StableLM-1.6B-4e-RePa-2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 18 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir" \
    --report_to wandb \
    --finetune_repa_mode $FINETUNE_REPA_MODE \
    --repa_gated_ratio $REPA_GATED_RATIO 
else
  echo "NODE_RANK=$NODE_RANK on host $THIS_HOST: skipping DeepSpeed launch."
fi