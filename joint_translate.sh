#!/bin/bash

# This script performs inference from a source language to a target language using fairseq model.


echo `date`
infname=$1          # path to the input file name
outfname=$2         # path to the output file name
src_lang=$3         # source language (according to the flores code)
tgt_lang=$4         # target language (according to the flores code)
ckpt_dir=$5         # path to the checkpoint directory


# check if the source language text requires transliteration
src_transliterate="true"
if [[ $src_lang == *"Arab"* ]] || [[ $src_lang == *"Olck"* ]] || \
    [[ $src_lang == *"Mtei"* ]] || [[ $src_lang == *"Latn"* ]]; then
    src_transliterate="false"
fi


# check if the target language text requires transliteration
tgt_transliterate="true"
if [[ $tgt_lang == *"Arab"* ]] || [[ $tgt_lang == *"Olck"* ]] || \
    [[ $tgt_lang == *"Mtei"* ]] || [[ $tgt_lang == *"Latn"* ]]; then
    tgt_transliterate="false"
fi


# define the prefixes for source and target languages
SRC_PREFIX='SRC'
TGT_PREFIX='TGT'


echo "Normalizing punctuations"
bash normalize_punctuation.sh $src_lang < $infname > $outfname._norm

echo "Adding do not translate tags"
python3 scripts/normalize_regex_inference.py $outfname._norm $outfname.norm
rm -rf $outfname._norm && mv $outfname.norm $outfname._norm

echo "Applying normalization and script conversion"
input_size=`python scripts/preprocess_translate.py $outfname._norm $outfname.norm $src_lang $src_transliterate false`
echo "Number of sentences in input: $input_size"


echo "Applying sentence piece"
spm_encode --model $ckpt_dir/vocab/model.SRC \
    --output_format=piece \
    < $outfname.norm \
    > $outfname._bpe

echo "Adding language tags"
python scripts/add_tags_translate.py $outfname._bpe $outfname.bpe $src_lang $tgt_lang


echo "Decoding"
fairseq-interactive $ckpt_dir/final_bin \
    -s $SRC_PREFIX -t $TGT_PREFIX \
    --distributed-world-size 1 --fp16 \
    --path $ckpt_dir/model/checkpoint_best.pt \
    --task translation \
    --user-dir model_configs \
    --skip-invalid-size-inputs-valid-test \
    --batch-size 128 --buffer-size 2500 --beam 5 \
    --input $outfname.bpe > $outfname.log 2>&1


echo "Extracting translations, script conversion and detokenization"
# this script extracts the translations, convert script from devanagari script to target language if needed and detokenizes the output
python scripts/postprocess_translate.py $outfname.log $outfname $input_size $tgt_lang $tgt_transliterate $ckpt_dir/vocab/model.TGT

echo "Translation completed"
