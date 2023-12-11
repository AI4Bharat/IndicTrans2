#!/bin/bash

data_dir=$1
teacher_ckpt_path=$2
wandb_project=${2:-"IT2_distillation"}

# use a learning rate of 1e-3 for Indic-En and 7e-4 for En-Indic

fairseq-train ${data_dir}/final_bin \
    --max-source-positions 256 \
    --max-target-positions 256 \
    --max-update 1000000 \
    --save-interval-updates 2500 \
    --arch transformer_base18L \
    --task translation_with_kd \
    --kd-strategy word_level \
    --share-decoder-input-output-embed \
    --teacher-checkpoint-path $teacher_ckpt_path \
    --criterion label_smoothed_cross_entropy_with_kd \
    --source-lang SRC \
    --target-lang TGT \
    --lr-scheduler inverse_sqrt \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-07 \
    --lr 1e-3 \
    --warmup-updates 4000 \
    --dropout 0.2 \
    --save-dir ${data_dir}/base18L \
    --save-interval 1 \
    --keep-interval-updates 5 \
    --no-epoch-checkpoints \
    --patience 10 \
    --skip-invalid-size-inputs-valid-test \
    --memory-efficient-fp16 \
    --update-freq 16 \
    --distributed-world-size 4 \
    --num-workers 24 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --wandb-project $wandb_project
