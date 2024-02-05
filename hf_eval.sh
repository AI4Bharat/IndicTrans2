#!/bin/bash

in_dir=$1

for devtest_dir in "${in_dir}/IN22/conv" "${in_dir}/IN22/gen" "${in_dir}/flores22"; do 
    for quant in "4bit" "8bit" "none"; do
        python hf_eval.py \
            --model "ai4bharat/indictrans2-indic-en-1B" \
            --quantization ${quant} \
            --devtest_dir ${devtest_dir}

        python hf_eval.py \
            --model "ai4bharat/indictrans2-indic-en-dist-200M" \
            --quantization ${quant} \
            --devtest_dir ${devtest_dir}

        python hf_eval.py \
            --model "ai4bharat/indictrans2-en-indic-1B" \
            --quantization ${quant} \
            --devtest_dir ${devtest_dir}

        python hf_eval.py \
            --model "ai4bharat/indictrans2-en-indic-dist-200M" \
            --quantization ${quant} \
            --devtest_dir ${devtest_dir}
    done 
done

for devtest_dir in "${in_dir}/IN22/conv" "${in_dir}/IN22/gen" "${in_dir}/flores22"; do 
    for pair in ${devtest_dir}/*; do
        pair_lang=$(basename $pair)
        src_lang=$(echo "$pair_lang" | cut -d "-" -f 1)
        tgt_lang=$(echo "$pair_lang" | cut -d "-" -f 2)
        
        if [[ $tgt_lang == "eng_Latn" ]]; then
            lang=$src_lang
        else
            lang=$tgt_lang
        fi

        all_models=""

        for model_name in "it2-1B" "it2-dist-200M"; do
            for quant in "4bit" "8bit" "none"; do
                system_name=${model_name}-${quant}-hf
                bash compute_metrics.sh ${pair}/test.${lang}.pred.${system_name} ${pair}/test.${lang} $lang > ${pair}/score.${system_name}.eng_Latn-${lang}.json
                bash compute_metrics.sh ${pair}/test.eng_Latn.pred.${system_name} ${pair}/test.eng_Latn eng_Latn > ${pair}/score.${system_name}.${lang}-eng_Latn.json
                all_models="${all_models},${system_name}"
            done
        done 
    done

    python scripts/collate_multi_system_metrics.py \
        --devtest_dir ${devtest_dir} \
        --systems ${all_models}
done 