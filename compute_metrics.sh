#!/bin/bash

# This script compute the evaluation metrics such as BLEU, chrF, chrF++ using the 
# detokenized predictions of the translation systems using sacrebleu (version 2.3.1).
# If the target language is:
#   English: directly use Moses tokenizer that is internally supported (`mteval-v13a`)
#   Indic: use IndicNLP tokenizers and skip tokenization step in sacrebleu.


echo `date`
pred_fname=$1       # path to the predction file
ref_fname=$2        # path to the reference file
tgt_lang=$3         # target language


if [ $tgt_lang == 'eng_Latn' ]; then
    # directly tokenize the prediction and reference files using sacrebleu and compute the metric
    sacrebleu $ref_fname < $pred_fname -m bleu chrf
    sacrebleu $ref_fname < $pred_fname -m chrf --chrf-word-order 2
else

    # indicnlp tokenize prediction and reference files before evaluation
    input_size=`python scripts/preprocess_translate.py $ref_fname $ref_fname.tok $tgt_lang false false`
    input_size=`python scripts/preprocess_translate.py $pred_fname $pred_fname.tok $tgt_lang false false`

    # since we are tokenizing with indicnlp separately, we are setting tokenize to none here
    sacrebleu --tokenize none $ref_fname.tok < $pred_fname.tok -m bleu chrf
    sacrebleu --tokenize none $ref_fname.tok < $pred_fname.tok -m chrf --chrf-word-order 2
fi
