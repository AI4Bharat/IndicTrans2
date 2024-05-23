#!/bin/bash

echo `date`
infname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
ckpt_dir=$5
model=$6

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

echo "Applying normalization and script conversion"
parallel --pipe --keep-order bash scripts/normalize_punctuation.sh $src_lang < $infname > $outfname._norm

input_size=$(grep -c '.' $outfname._norm)
echo "Input size: ${input_size}"

parallel --pipe --keep-order python scripts/normalize_regex_inference.py < $outfname._norm > $outfname.norm

mv $outfname.norm $outfname._norm

parallel --pipe --keep-order python scripts/preprocess_translate.py $src_lang $src_transliterate false < $outfname._norm > $outfname.norm 

echo "Applying sentence piece"
parallel --pipe --keep-order spm_encode --model $ckpt_dir/vocab/model.SRC --output_format=piece < $outfname.norm > $outfname._bpe

echo "Adding Tags"
parallel --pipe --keep-order bash scripts/add_tags_translate.sh $src_lang $tgt_lang < $outfname._bpe > $outfname.bpe 


echo "Decoding"
fairseq-interactive ${ckpt_dir}/final_bin \
    --path ${ckpt_dir}/${model}/checkpoint_best.pt \
    --source-lang SRC \
    --target-lang TGT \
    --batch-size 256 \
    --buffer-size 2500 \
    --beam 5 \
    --memory-efficient-fp16 \
    --num-workers 24 \
    --force-override-max-positions "(4096, 4096)" \
    --input $outfname.bpe > $outfname.log 2>&1

echo "Extracting translations, script conversion and detokenization"
# this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.
python scripts/postprocess_translate.py $outfname.log $outfname $input_size $tgt_lang $tgt_transliterate $ckpt_dir/vocab/model.TGT

# Purge the intermediate files to declutter the directory.
# But keep the log files for future reference
echo "Purging intermediate files"
rm $outfname.bpe $outfname.norm $outfname._bpe $outfname._norm

echo -e "Translation completed\n\n"