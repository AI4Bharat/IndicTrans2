#!/bin/bash

# This script performs significance testing for metrics such as BLEU, chrF++ using sacrebleu on the evaluation set
# where each subdirectory contains En-X pair


echo `date`
devtest_data_dir=$1                         # path to the evaluation directory

# we predefine a set of systems which we consider for evaluation
# feel free to change the below line in case you want to add or remove any system
system=(google azure nllb mbart50 m2m100 itv1 itv2)


# get a list of language pairs in the `devtest_data_dir`
pairs=$(ls -d $devtest_data_dir/eng_Latn-* | sort)


# iterate over each language pair
for pair in ${pairs[@]}; do
    # extract the source and target languages from the pair name
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)

    if [[ $src_lang == "eng_Latn" ]]; then
    
        # ----------------------------------------------------------------------
        #                           en - indic direction
        # ----------------------------------------------------------------------
        echo "${src_lang} - ${tgt_lang}"

        # find all the prediction files for different systems and tokenize it using IndicNLP
        pred_fnames=$devtest_data_dir/$pair/test.${tgt_lang}.pred.*
        ref_fname=$devtest_data_dir/$pair/test.${tgt_lang}

        for pred_fname in $(find . -type f -name $pred_fnames); do
            input_size=`python scripts/preprocess_translate.py $pred_fname $pred_fname.tok $tgt_lang false false`
        done

        input_size=`python scripts/preprocess_translate.py $ref_fname $ref_fname.tok $tgt_lang false false`

        ref_fname=$devtest_data_dir/$pair/test.${tgt_lang}.tok
        itv2_fname=$devtest_data_dir/$pair/test.${tgt_lang}.pred.itv2.tok
        sys_fnames=$devtest_data_dir/$pair/test.${tgt_lang}.pred.*.tok
        bleu_out_fname=$devtest_data_dir/$pair/${src_lang}_${tgt_lang}_bleu_significance.txt
        chrF_out_fname=$devtest_data_dir/$pair/${src_lang}_${tgt_lang}_chrF++_significance.txt

        sacrebleu --tokenize none $ref_fname -i $itv2_fname $sys_fnames --paired-bs -m bleu --format text > $bleu_out_fname
        sacrebleu --tokenize none $itv2_fname $sys_fnames --paired-bs -m chrf --chrf-word-order 2 --format text > $chrF_out_fname

        # ----------------------------------------------------------------------
        #                           indic - en direction
        # ----------------------------------------------------------------------
        echo "${tgt_lang} - ${src_lang}"

        ref_fname=$devtest_data_dir/$pair/test.${src_lang}
        itv2_fname=$devtest_data_dir/$pair/test.${src_lang}.pred.itv2
        sys_fnames=$devtest_data_dir/$pair/test.${src_lang}.pred.*
        bleu_out_fname=$devtest_data_dir/$pair/${tgt_lang}_${src_lang}_bleu_significance.txt
        chrF_out_fname=$devtest_data_dir/$pair/${tgt_lang}_${src_lang}_chrF++_significance.txt

        sacrebleu --tokenize none $ref_fname -i $itv2_fname $sys_fnames --paired-bs -m bleu --format text > $bleu_out_fname
        sacrebleu --tokenize none $itv2_fname $sys_fnames --paired-bs -m chrf --chrf-word-order 2 --format text > $chrF_out_fname
    
    fi
