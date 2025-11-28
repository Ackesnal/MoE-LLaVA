#!/bin/bash
#SBATCH --job-name=eval_repa_moe_vqav2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=5-9%4
#SBATCH --output=logs/eval_repa_moe_vqav2_%A_%a.out
#SBATCH --error=logs/eval_repa_moe_vqav2_%A_%a.err

# Modules (matching training environment)
module load gcc/12.3.0
module load cuda/12.4.0
module load cudnn/9.3.0-cu12
module load miniconda3/23.5.2
module load ninja/1.11.1
module load sqlite/3.43.1
module load nccl/2.20.5-cu124

# Work directory
cd /home/li309/pct_code/moe/MoE-LLaVA

# Activate environment
source activate /home/li309/pct_code/venv/moellava-test2

# Derive gated ratio from array index (1-9 -> 0.1-0.9)
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[WARN] SLURM_ARRAY_TASK_ID not set; defaulting ratio=0.5" >&2
    GATED_RATIO=0.5
else
    GATED_RATIO=$(awk "BEGIN {printf \"%.1f\", ${SLURM_ARRAY_TASK_ID}/10}")
fi
GATED_RATIO_TAG=${GATED_RATIO/./p}
echo "Evaluating checkpoint with GATED_RATIO=${GATED_RATIO} (tag=${GATED_RATIO_TAG})"


gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CONV="stablelm"
CKPT_NAME="MoE-LLaVA-StableLM-1.6B-4e-RePa-Only_MoE-Dual_Branch-Full_Model-ratio${GATED_RATIO_TAG}"
CKPT="finetuned_checkpoints/${CKPT_NAME}"
SPLIT="llava_vqav2_mscoco_test2015"
EVAL="/scratch3/li309/data/llava_data/eval"


# Basic existence check
if [ ! -d "${CKPT}" ]; then
    echo "[ERROR] Checkpoint directory not found: ${CKPT}" >&2
    exit 1
fi


for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) moellava/eval/model_vqa_loader.py \
        --model-path ${CKPT} \
        --question-file ${EVAL}/vqav2/$SPLIT.jsonl \
        --image-folder ${EVAL}/vqav2/test2015 \
        --answers-file ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV} &
done

wait

output_file=${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3 scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt ${CKPT_NAME} --dir ${EVAL}/vqav2