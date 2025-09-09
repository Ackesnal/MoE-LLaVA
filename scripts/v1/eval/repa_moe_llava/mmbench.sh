#!/bin/bash

SPLIT="mmbench_dev_20230712"

CONV="stablelm"
CKPT_NAME="MoE-LLaVA-StableLM-1.6B-4e"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/mnt/data/llava_data/eval"
deepspeed moellava/eval/model_vqa_mmbench.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/mmbench/$SPLIT.tsv \
    --answers-file ${EVAL}/mmbench/answers/$SPLIT/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV}

mkdir -p ${EVAL}/mmbench/answers_upload/$SPLIT

python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file ${EVAL}/mmbench/$SPLIT.tsv \
    --result-dir ${EVAL}/mmbench/answers/$SPLIT \
    --upload-dir ${EVAL}/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT_NAME}


