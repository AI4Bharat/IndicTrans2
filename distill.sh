#!/bin/bash

data_dir=$1

/nlsasfs/home/ai4bharat/yashkm/miniconda3/envs/itd/bin/fairseq-train ${data_dir}/final_bin \
    --max-source-positions 256 \
    --max-target-positions 256 \
    --max-update 1000000 \
    --save-interval-updates 2500 \
    --arch transformer_base18L \
    --task translation_with_kd \
    --kd-strategy word_level \
    --share-decoder-input-output-embed \
    --teacher-checkpoint-path ${data_dir}/model/checkpoint_best.pt \
    --criterion label_smoothed_cross_entropy_with_kd \
    --source-lang SRC \
    --target-lang TGT \
    --lr-scheduler inverse_sqrt \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-07 \
    --lr 7e-4 \
    --warmup-updates 4000 \
    --dropout 0.2 \
    --save-dir ${data_dir}/base18L_dec_shared_embed \
    --save-interval 1 \
    --keep-interval-updates 1 \
    --no-epoch-checkpoints \
    --patience 10 \
    --skip-invalid-size-inputs-valid-test \
    --memory-efficient-fp16 \
    --update-freq 1 \
    --distributed-world-size 8 \
    --num-workers 32 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args "{\"beam\": 5, \"lenpen\": 1.0, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric
