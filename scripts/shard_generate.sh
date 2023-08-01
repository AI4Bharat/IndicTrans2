#!/bin/bash

# mainly for top-11 languages

exp_dir=$1
model=$2
src_lang=$3
tgt_lang=$4
shard_id=$5

echo `date`

outfname=$exp_dir/binarized_train_data_only/${src_lang}_${tgt_lang}_p${shard_id}_final_bin/train.${tgt_lang}_p${shard_id}.log
out_hyp=$exp_dir/binarized_train_data_only/${src_lang}_${tgt_lang}_p${shard_id}_final_bin/train.${tgt_lang}_p${shard_id}.hyp
data_bin=$exp_dir/binarized_train_data_only/${src_lang}_${tgt_lang}_p${shard_id}_final_bin

fairseq-generate $data_bin \
    --source-lang SRC \
    --target-lang TGT \
    --num-workers 24 \
    --memory-efficient-fp16 \
    --path $exp_dir/$model/checkpoint_best.pt \
    --gen-subset train \
    --batch-size 128 \
    --beam 5 \
    --skip-invalid-size-inputs-valid-test > $outfname 2>&1

value=`cat $exp_dir/binarized_train_data_only/${src_lang}_${tgt_lang}_p${shard_id}_final_bin/preprocess.log | grep train.SRC`
array=($value)
input_size=${array[2]}
echo "Number of sentences: $input_size"

echo "Extracting translations, script conversion and detokenization"
# this part reverses the transliteration from devnagiri script to source lang and then detokenizes it.
python postprocess_translate.py $outfname $out_hyp $input_size $tgt_lang true $exp_dir/vocab/model.TGT