#!/bin/bash

pred_fname=$1
ref_fname=$2
tgt_lang=$3

# This script compute the evaluation metrics such as BLEU, chrF, chrF++ using the 
# detokenized predictions of the translation systems using sacrebleu (version 2.3.1).
# If the target language is:
#   English: directly use Moses tokenizer that is internally supported (`mteval-v13a`)
#   Indic: use IndicNLP tokenizers and skip tokenization step in sacrebleu.


if [ $tgt_lang == 'eng_Latn' ]; then
    # indic to en models
    sacrebleu $ref_fname < $pred_fname -m bleu chrf --chrf-word-order 2
else
    # indicnlp tokenize predictions and reference files before evaluation
    parallel --pipe --keep-order python scripts/preprocess_translate.py $tgt_lang false false < $ref_fname > $ref_fname.tok 
    parallel --pipe --keep-order python scripts/preprocess_translate.py $tgt_lang false false < $pred_fname > $pred_fname.tok 

    # since we are tokenizing with indicnlp separately, we are setting tokenize to none here
    sacrebleu --tokenize none $ref_fname.tok < $pred_fname.tok -m bleu chrf --chrf-word-order 2

    # purge intermediate files
    rm $ref_fname.tok $pred_fname.tok
fi
