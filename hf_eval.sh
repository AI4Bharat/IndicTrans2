#!/bin/bash

in_dir=${1:-"../distillation_data"}

for direction in "indic-en" "en-indic"; do
    for devtest_dir in "${in_dir}/IN22/conv" "${in_dir}/IN22/gen" "${in_dir}/flores22"; do 
        for model in "ai4bharat/indictrans2-${direction}-1B" "ai4bharat/indictrans2-${direction}-dist-200M"; do
            for quantization in "4bit" "8bit" "none"; do
                echo " | > Starting evaluation with on ${devtest_dir} ${direction} for ${model} with ${quantization} quantization ..."
                python hf_eval.py \
                    --direction $direction \
                    --model $model \
                    --devtest_dir $devtest_dir \
                    --quantization $quantization
            done 
        done 
    done 
done 