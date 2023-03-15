#!/bin/bash

exp_dir=$1
data_dir=$2
bpe_dir=$3
src_lang=$4
tgt_lang=$5
split=$6

in_split_dir=$data_dir/$split
out_split_dir=$bpe_dir/$split

echo "Apply Sentence Piece tokenization to SRC corpus"

# for very large datasets, use gnu-parallel to speed up applying bpe
# uncomment the below line if the apply bpe is slow

parallel --pipe --keep-order \
spm_encode --model=$exp_dir/vocab/model.SRC \
    --output_format=piece \
    < $in_split_dir.$src_lang \
    > $out_split_dir.$src_lang

echo "Apply Sentence Piece tokenization to TGT corpus"

# for very large datasets, use gnu-parallel to speed up applying bpe
# uncomment the below line if the apply bpe is slow

parallel --pipe --keep-order \
spm_encode --model=$exp_dir/vocab/model.TGT \
    --output_format=piece \
    < $in_split_dir.$tgt_lang \
    > $out_split_dir.$tgt_lang
