#!/bin/bash
#SBATCH --job-name=eval_repa_moe_scienceqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=1-9%4
#SBATCH --output=logs/eval_repa_moe_scienceqa_%A_%a.out
#SBATCH --error=logs/eval_repa_moe_scienceqa_%A_%a.err

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

CONV="stablelm"
CKPT_NAME="MoE-LLaVA-StableLM-1.6B-4e-RePa-Save-Experiment-ratio${GATED_RATIO_TAG}"
CKPT="finetuned_checkpoints/${CKPT_NAME}"
EVAL="/scratch3/li309/data/llava_data/eval"

# Basic existence check
if [ ! -d "${CKPT}" ]; then
    echo "[ERROR] Checkpoint directory not found: ${CKPT}" >&2
    exit 1
fi

ANS_DIR="${EVAL}/scienceqa/answers"
mkdir -p "${ANS_DIR}"

ANS_FILE="${ANS_DIR}/${CKPT_NAME}.jsonl"
OUT_JSONL="${ANS_DIR}/${CKPT_NAME}_output.jsonl"
RES_JSON="${ANS_DIR}/${CKPT_NAME}_result.json"

echo "CKPT: ${CKPT}"
echo "Answer file: ${ANS_FILE}"

# Run VQA-style evaluation (generation)
deepspeed moellava/eval/model_vqa_science.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
    --image-folder "${EVAL}/scienceqa/images/test" \
    --answers-file "${ANS_FILE}" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode "${CONV}"

# Post evaluation scoring
python3 moellava/eval/eval_science_qa.py \
    --base-dir "${EVAL}/scienceqa" \
    --result-file "${ANS_FILE}" \
    --output-file "${OUT_JSONL}" \
    --output-result "${RES_JSON}"

echo "Done: ratio ${GATED_RATIO} -