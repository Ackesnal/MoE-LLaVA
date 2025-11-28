#!/bin/bash
#SBATCH --job-name=eval_repa_moe_scienceqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=5-9%4
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
CKPT_NAME="MoE-LLaVA-StableLM-1.6B-4e-RePa-Only_MoE-Dual_Branch-Full_Model-ratio${GATED_RATIO_TAG}"
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

# Run VQA-style evaluation (generation)
deepspeed --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR moellava/eval/model_vqa_science.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
    --image-folder "${EVAL}/scienceqa/images/test" \
    --answers-file "${ANS_FILE}" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode "${CONV}" \
    --repa_gated_ratio ${GATED_RATIO}

# Post evaluation scoring
python3 moellava/eval/eval_science_qa.py \
    --base-dir "${EVAL}/scienceqa" \
    --result-file "${ANS_FILE}" \
    --output-file "${OUT_JSONL}" \
    --output-result "${RES_JSON}"

echo "Done: ratio ${GATED_RATIO} -