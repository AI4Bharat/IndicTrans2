#!/bin/bash

echo `date`
exp_dir=$1
vocab_dir=$2
src_lang=$3
tgt_lang=$4
num_shards=$5
train_data_dir=$6

root=$(dirname $0)

echo "Running data preparation ${exp_dir}"

pair=$src_lang-$tgt_lang
echo "$src_lang-$tgt_lang"

SRC_PREFIX='SRC'
TGT_PREFIX='TGT'

infname=$train_data_dir/$pair/train.$src_lang

input_size=$(awk 'END{print NR}' $infname)
echo "Number of sentences: $input_size"

window_size=$((input_size / num_shards))
echo "Number of sentences in each shard: $window_size"

num_examples=1
shard_id=0

while [[ $num_examples -le $input_size ]]; do
    start=$num_examples
    num_examples=$((num_examples + window_size))
    end="$((num_examples > input_size ? input_size : num_examples))"
    echo "$start - $end"

    shard_infname=$infname.p${shard_id}
    out_data_dir=$exp_dir/binarized_train_data_only/${src_lang}_${tgt_lang}_p${shard_id}_final
    
    sed -n -e "$start,$end p" -e "$end q" $infname > $shard_infname

    echo "Applying normalization and script conversion"

    echo "Normalizing punctuations ..."
    parallel --pipe --keep-order bash normalize_punctuation.sh $src_lang < $shard_infname > $shard_infname._norm

    echo "Applying do not translate tags ..."
    parallel --pipe --keep-order python normalize_regex_inference.py < $shard_infname._norm > $shard_infname.norm

    echo "Removing redundant files and renaming ..."
    mv $shard_infname.norm $shard_infname._norm

    echo "Preprocess transliteration ..."
    parallel --pipe --keep-order python preprocess_translate.py $src_lang true false < $shard_infname._norm > $shard_infname.norm 
    
    shard_input_size=$(grep -c '.' $shard_infname.norm)
    echo "Number of sentences in input: $shard_input_size"

    echo "Applying sentence piece ..."
    parallel --pipe --keep-order spm_encode --model $vocab_dir/vocab/model.SRC --output_format=piece < $shard_infname.norm > $shard_infname._bpe

    echo "Adding language tags ..."
    parallel --pipe --keep-order add_tags_translate.sh $src_lang $tgt_lang < $outfname._bpe > $outfname.bpe

    mkdir -p $out_data_dir

    src_infname=$shard_infname.bpe
    src_outfname=$out_data_dir/train.SRC
    tgt_outfname=$out_data_dir/train.TGT
    outfname=$out_data_dir/train.$src_lang.p${shard_id}

    # Copy the processed files for binarization
    cp -r $src_infname $src_outfname
    cp -r $src_infname $tgt_outfname
    cp -r $shard_infname $outfname

    num_examples=$((num_examples + 1))
    shard_id=$((shard_id + 1))

    echo "Binarizing data ..."
    # Binarize the training data for using with fairseq train

    fairseq-preprocess \
        --source-lang SRC \
        --target-lang TGT \
        --trainpref $out_data_dir/train \
        --destdir ${out_data_dir}_bin \
        --srcdict $vocab_dir/final_bin/dict.SRC.txt \
        --tgtdict $vocab_dir/final_bin/dict.TGT.txt \
        --workers 24

    rm -rf $out_data_dir

done