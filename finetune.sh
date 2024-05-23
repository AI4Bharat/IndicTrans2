data_dir=$1
restore_from_dir=$2
wandb_project=${3:-"fairseq"}

fairseq-train ${data_dir}/final_bin \
    --task translation \
    --max-source-positions 256 \
    --max-target-positions 256 \
    --max-update 1000000 \
    --save-interval 1 \
    --save-interval-updates 1000 \
    --arch transformer_IT2_dist \
    --criterion label_smoothed_cross_entropy \
    --share-decoder-input-output-embed \
    --source-lang SRC \
    --target-lang TGT \
    --lr-scheduler inverse_sqrt \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 2000 \
    --dropout 0.2 \
    --restore-file ${restore_from_dir}/checkpoint_best.pt \
    --save-dir ${restore_from_dir}_seed_ft \
    --patience 10 \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints \
    --keep-interval-updates 1 \
    --update-freq 1 \
    --distributed-world-size 8 \
    --max-tokens 1024 \
    --memory-efficient-fp16 \
    --lr 3e-5 \
    --num-workers 24 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe sentencepiece \
    --maximize-best-checkpoint-metric \
    --best-checkpoint-metric bleu \
    --reset-dataloader \
    --reset-lr-scheduler \
    --reset-optimizer \
    --wandb-project $wandb_project