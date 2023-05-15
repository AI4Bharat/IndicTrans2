#!/bin/bash

echo `date`
infname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
ckpt_dir=$5


src_transliterate="true"
if [[ $src_lang == *"Arab"* ]] || [[ $src_lang == *"Olck"* ]] || \
    [[ $src_lang == *"Mtei"* ]] || [[ $src_lang == *"Latn"* ]]; then
    src_transliterate="false"
fi

tgt_transliterate="true"
if [[ $tgt_lang == *"Arab"* ]] || [[ $tgt_lang == *"Olck"* ]] || \
    [[ $tgt_lang == *"Mtei"* ]] || [[ $tgt_lang == *"Latn"* ]]; then
    tgt_transliterate="false"
fi


SRC_PREFIX='SRC'
TGT_PREFIX='TGT'


echo "Applying normalization and script conversion"

bash normalize_punctuation.sh $src_lang < $infname > $outfname._norm

echo "Applying do not translate tags for dev"
python3 scripts/normalize_regex_inference.py $outfname._norm $outfname.norm
rm -rf $outfname._norm && mv $outfname.norm $outfname._norm

input_size=`python scripts/preprocess_translate.py $outfname._norm $outfname.norm $src_lang $src_transliterate false`
echo "Number of sentences in input: $input_size"


echo "Applying sentence piece"
spm_encode --model $ckpt_dir/vocab/model.SRC \
    --output_format=piece \
    < $outfname.norm \
    > $outfname._bpe

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
# this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.
python scripts/postprocess_translate.py $outfname.log $outfname $input_size $tgt_lang $tgt_transliterate $ckpt_dir/vocab/model.TGT

echo "Translation completed"
