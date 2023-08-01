#/bin/bash

echo `date`

exp_dir=$1          # path to the experiment directory
data_dir=$2         # path to the data directory where all lang pairs are concatenated
bpe_dir=$3          # path to the tokenized data directory
src_lang=$4         # source language
tgt_lang=$5         # target language
split=$6            # name of the split


in_split_dir=$data_dir/$split
out_split_dir=$bpe_dir/$split


echo "Applying Sentence Piece tokenization to SRC corpus"
parallel --pipe --keep-order spm_encode --model=$exp_dir/vocab/model.SRC --output_format=piece < $in_split_dir.$src_lang > $out_split_dir.$src_lang


echo "Applying Sentence Piece tokenization to TGT corpus"
parallel --pipe --keep-order spm_encode --model=$exp_dir/vocab/model.TGT --output_format=piece < $in_split_dir.$tgt_lang > $out_split_dir.$tgt_lang