export CUDA_VISIBLE_DEVICES=0

data_dir=${1:-"en-indic-exp"}
model_name=${2:-"ai4bharat/indictrans2-en-indic-dist-200M"}
output_dir=${3:-"output"}
direction=${4:-"en-indic"}
src_lang_list=${5:-"eng_Latn"}
tgt_lang_list=${6:-"asm_Beng,ben_Beng,guj_Gujr,hin_Deva,kan_Knda,mal_Mlym,mar_Deva,npi_Deva,ory_Orya,pan_Guru,tam_Taml,tel_Telu,urd_Arab"}

python3 train_lora.py \
    --data_dir $data_dir \
    --model_name $model_name \
    --output_dir $output_dir \
    --direction $direction \
    --src_lang_list $src_lang_list \
    --tgt_lang_list $tgt_lang_list \
    --save_steps 1000 \
    --max_steps 1000000 \
    --batch_size 128 \
    --grad_accum_steps 2 \
    --warmup_steps 4000 \
    --max_grad_norm 1.0 \
    --learning_rate 3e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --optimizer adamw_torch \
    --lr_scheduler inverse_sqrt \
    --num_workers 16 \
    --metric_for_best_model eval_BLEU \
    --greater_is_better \
    --mixed_precision fp16 \
    --patience 10 \
    --label_smoothing 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj" \
    --lora_dropout 0.05 \
    --lora_r 8 \
    --lora_alpha 32 \
    --generation_config '{"max_new_tokens": 256, "min_length": 1, "num_beams": 5, "use_cache": true, "num_return_sequences": 1}'
