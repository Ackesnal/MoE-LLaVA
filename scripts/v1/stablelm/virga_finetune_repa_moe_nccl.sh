#!/bin/bash
#SBATCH --job-name=finetune_repa_moe_nccl
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=5:00:00
#SBATCH --output=logs/finetune_repa_moe_nccl_%A_%a.out
#SBATCH --error=logs/finetune_repa_moe_nccl_%A_%a.err

#SBATCH --array=1-9

# Load required modules
module load gcc/12.3.0
module load cuda/12.4.0
module load cudnn/9.3.0-cu12
module load miniconda3/23.5.2
module load ninja/1.11.1
module load sqlite/3.43.1
module load nccl/2.20.5-cu124

# Change to working directory
cd /home/li309/pct_code/moe/MoE-LLaVA

# Create logs directory if it doesn't exist
mkdir -p logs

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
# Derive gated ratio from Slurm array index: 1-9 -> 0.1-0.9
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[WARN] SLURM_ARRAY_TASK_ID not set; defaulting GATED_RATIO=0.5" >&2
    GATED_RATIO=0.5
else
    GATED_RATIO=$(awk "BEGIN {printf \"%.1f\", ${SLURM_ARRAY_TASK_ID}/10}")
fi
echo "Using GATED_RATIO=${GATED_RATIO} from SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2

# Build per-run output directory incorporating gated ratio (sanitize decimal point)
GATED_RATIO_TAG=${GATED_RATIO/./p}
OUTPUT_DIR=./finetuned_checkpoints/MoE-LLaVA-StableLM-1.6B-4e-RePa-Save-Experiment-ratio${GATED_RATIO_TAG}
echo "OUTPUT_DIR: ${OUTPUT_DIR}" >&2

# Redirect logs to ratio-tagged files (after we know GATED_RATIO)
RATIO_LOG_PREFIX="logs/finetune_repa_moe_nccl_ratio${GATED_RATIO_TAG}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
echo "Redirecting logs to ${RATIO_LOG_PREFIX}.out/.err" >&2
exec > >(tee -a ${RATIO_LOG_PREFIX}.out)
exec 2> >(tee -a ${RATIO_LOG_PREFIX}.err >&2)

# Set data paths
JSON_FOLDER="/scratch3/li309/data/llava_data/train_json"
IMAGE_FOLDER="/scratch3/li309/data/llava_data/train_data"

# NCCL environment variables for multi-node communication
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG=WARN

function get_free_port() {
    # Function to find a free port
    local port
    local max_attempts=10
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
        # Check if port is truly available
        if ! ss -tuln | grep -q ":$port "; then
            echo $port
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    # Fallback to a random port in high range
    echo $((29500 + RANDOM % 1000))
}

# Set distributed training environment variables
export MASTER_PORT=$(get_free_port)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo "VERSION: 1.6 (NCCL-only)"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"

# Use torchrun approach to avoid MPI dependency completely
srun --export=ALL bash -c "
WANDB_MODE=offline HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=\$SLURM_GPUS_ON_NODE \
    --rdzv_id=\$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
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
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 18 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ./cache_dir \
    --report_to wandb \
    --finetune_repa_mode $FINETUNE_REPA_MODE \
    --gated_ratio $GATED_RATIO
"
