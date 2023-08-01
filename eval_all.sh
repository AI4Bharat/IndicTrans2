# !/bin/bash

rm -rf ../distillation_data/IN22/conv/*.csv
bash eval.sh ../distillation_data/IN22/conv ../distillation_data/BPCC0_indic_en_bin base18L_7e-4 itv2_1
bash eval.sh ../distillation_data/IN22/conv ../distillation_data/BPCC0_indic_en_bin base18L_7e-4_seed_ft itv2_2

rm -rf ../distillation_data/IN22/gen/*.csv
bash eval.sh ../distillation_data/IN22/gen ../distillation_data/BPCC0_indic_en_bin base18L_7e-4 itv2_1
bash eval.sh ../distillation_data/IN22/gen ../distillation_data/BPCC0_indic_en_bin base18L_7e-4_seed_ft itv2_2

rm -rf ../distillation_data/flores22/*.csv
bash eval.sh ../distillation_data/flores22 ../distillation_data/BPCC0_indic_en_bin base18L itv2_1
bash eval.sh ../distillation_data/flores22 ../distillation_data/BPCC0_indic_en_bin base18L_seed_ft itv2_2