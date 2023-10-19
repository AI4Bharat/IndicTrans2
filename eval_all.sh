#!/bin/bash

# evaluate the distilled models
bash eval.sh ../distillation_data/IN22/conv ../distillation_data/BPCC0_indic_en_bin base18L_avg_dec_shared_embed it2-dist
bash eval.sh ../distillation_data/IN22/gen ../distillation_data/BPCC0_indic_en_bin base18L_avg_dec_shared_embed it2-dist
bash eval.sh ../distillation_data/flores22 ../distillation_data/BPCC0_indic_en_bin base18L_avg_dec_shared_embed it2-dist
# evaluate the distilled seed ft models
bash eval.sh ../distillation_data/IN22/conv ../distillation_data/BPCC0_indic_en_bin base18L_avg_dec_shared_embed_seed_ft it2-dist-seed-ft
bash eval.sh ../distillation_data/IN22/gen ../distillation_data/BPCC0_indic_en_bin base18L_avg_dec_shared_embed_seed_ft it2-dist-seed-ft
bash eval.sh ../distillation_data/flores22 ../distillation_data/BPCC0_indic_en_bin base18L_avg_dec_shared_embed_seed_ft it2-dist-seed-ft