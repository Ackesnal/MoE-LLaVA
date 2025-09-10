#!/bin/bash

CONV="qwen"
CKPT_NAME="MoE-LLaVA-Qwen-1.8B-4e-RePa-Save/checkpoint-40"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/mnt/data/llava_data/eval"
deepspeed moellava/eval/model_vqa_science.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
    --image-folder ${EVAL}/scienceqa/images/test \
    --answers-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV}

python3 moellava/eval/eval_science_qa.py \
    --base-dir ${EVAL}/scienceqa \
    --result-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --output-file ${EVAL}/scienceqa/answers/${CKPT_NAME}_output.jsonl \
    --output-result ${EVAL}/scienceqa/answers/${CKPT_NAME}_result.json
