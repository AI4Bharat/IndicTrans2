#!/bin/bash

mkdir -p logs
rm ../distillation_data/IN22/conv/*.csv ../distillation_data/IN22/gen/*.csv ../distillation_data/flores22/*.csv

echo -e "\nworking on itv2_1"
CUDA_VISIBLE_DEVICES=0 bash eval.sh ../distillation_data/IN22/conv ../distillation_data/BPCC0_indic_en_bin base18L_dec_shared_embed itv2_1 > logs/conv1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash eval.sh ../distillation_data/IN22/gen ../distillation_data/BPCC0_indic_en_bin base18L_dec_shared_embed itv2_1 > logs/gen1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash eval.sh ../distillation_data/flores22 ../distillation_data/BPCC0_indic_en_bin base18L_dec_shared_embed itv2_1 > logs/flores1.log 2>&1 &
wait

echo -e "\nworking on itv2_2"
CUDA_VISIBLE_DEVICES=0 bash eval.sh ../distillation_data/IN22/conv ../distillation_data/BPCC0_indic_en_bin base18L_dec_shared_embed_seed_ft itv2_2 > logs/conv2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash eval.sh ../distillation_data/IN22/gen ../distillation_data/BPCC0_indic_en_bin base18L_dec_shared_embed_seed_ft itv2_2 > logs/gen2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash eval.sh ../distillation_data/flores22 ../distillation_data/BPCC0_indic_en_bin base18L_dec_shared_embed_seed_ft itv2_2 > logs/flores2.log 2>&1 &
wait