export CUDA_VISIBLE_DEVICES=0

data_dir=${1:-"en-indic-exp"}
model_name=${2:-"ai4bharat/indictrans2-en-indic-dist-200M"}
output_dir=${3:-"output"}
src_lang_list=${4:-"eng_Latn"}
tgt_lang_list=${5:-"asm_Beng,ben_Beng,guj_Gujr,hin_Deva,kan_Knda,mal_Mlym,mar_Deva,npi_Deva,ory_Orya,pan_Guru,tam_Taml,tel_Telu,urd_Arab"}

python3 train_lora.py \
    --data_dir $data_dir \
    --model_name $model_name \
    --output_dir $output_dir \
    --src_lang_list $src_lang_list \
    --tgt_lang_list $tgt_lang_list \
    --save_steps 1000 \
    --max_steps 1000000 \
    --batch_size 32 \
    --grad_accum_steps 4 \
    --warmup_steps 4000 \
    --max_grad_norm 1.0 \
    --learning_rate 2e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --optimizer adamw_torch \
    --lr_scheduler inverse_sqrt \
    --num_workers 16 \
    --metric_for_best_model eval_BLEU \
    --greater_is_better \
    --patience 10 \
    --weight_decay 0.01 \
    --lora_target_modules "q_proj,k_proj" \
    --lora_dropout 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --print_samples