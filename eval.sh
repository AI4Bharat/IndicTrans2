#!/bin/bash

# This script evaluates the performance of a machine translation system 
# on a evaluation set in forward direction. For example, if the evaluation set 
# consists of language pairs, such as En-X, where En represents the English language 
# and X represents the target Indic language then this script accesses the translation
# system from the English language (En) to the target Indic language (X) direction.


echo `date`
devtest_data_dir=$1         # path to the evaluation directory
ckpt_dir=$2                 # path to the checkpoint directory
system=${3:-"it2"}          # name of the machine translation system


# get a list of language pairs in the `devtest_data_dir`
pairs=$(ls -d $devtest_data_dir/* | sort)


# iterate over each language pair
for pair in ${pairs[@]}; do
    # extract the source and target languages from the pair name
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)

    src_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$src_lang
    tgt_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang

    # check if the source and target files exists
    if [ -f "$src_fname" ] && [ -f "$tgt_fname" ]; then
        echo "Evaluating $src_lang-$tgt_lang ..."
    else 
        echo "Skipping $src_lang-$tgt_lang ..."
        continue
    fi

    # generate translations if the system name contains "it2"
    if [[ $system == *"it2"* ]]; then
        echo "Generating Translations"
        bash joint_translate.sh $src_fname $tgt_fname.pred.$system $src_lang $tgt_lang $ckpt_dir
    fi

    # compute automatic string-based metrics if the prediction exists for the system
    if [[ -f "${tgt_fname}.pred.${system}" ]]; then
        echo "Computing Metrics"
        bash compute_metrics.sh $tgt_fname.pred.$system $tgt_fname $tgt_lang > $devtest_data_dir/$src_lang-$tgt_lang/${src_lang}_${tgt_lang}_${system}_scores.txt
    fi

    # remove the intermediate files
    rm -rf $tgt_fname.pred.$system.*
    rm -rf $devtest_data_dir/$src_lang-$tgt_lang/*.tok

done
