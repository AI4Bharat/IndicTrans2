#!/bin/bash

# This script evaluates the performance of a machine translation system 
# on a evaluation set in forward direction. For example, if the evaluation set 
# consists of language pairs, such as X-Y, where X represents the source Indic language 
# and Y represents the target Indic language then this script accesses the translation
# system from the source Indic language (X) to the target Indic language (Y) direction 
# using English as the pivot language (X -> En and En -> Y).


echo `date`
devtest_data_dir=$1                 # path to the evaluation set
pivot_lang=${2:-"eng_Latn"}         # pivot language of choice
src2pivot_ckpt_dir=$3               # path to the Indic-En checkpoint directory
pivot2tgt_ckpt_dir=$4               # path of the En-Indic checkpoint directory
system=${3:-"itv2"}                 # name of the machine translation system


# get a list of language pairs in the `devtest_data_dir`
pairs=$(ls -d $devtest_data_dir/* | sort)


# iterate over each language pair
for pair in ${pairs[@]}; do
    # extract the source and target languages from the pair name
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)

    src_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$src_lang
    pivot_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$pivot_lang
    tgt_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang

    # check if the source and target files exists
    if [ -f "$src_fname" ] && [ -f "$tgt_fname" ]; then
        echo "Evaluating $src_lang-$tgt_lang ..."
    else 
        echo "Skipping $src_lang-$tgt_lang ..."
        continue
    fi

    # generate translations if the system name contains "itv2"
    if [[ $system == *"itv2"* ]]; then
        # source to pivot translation
        echo "Generating Source to Pivot Translations"
        bash joint_translate.sh $src_fname $pivot_fname.pred.$system $src_lang $pivot_lang $src2pivot_ckpt_dir
        
        # pivot to target translation
        echo "Generating Pivot to Target Translations"
        bash joint_translate.sh $pivot_fname.pred.$system $tgt_fname.pred.$system $pivot_lang $tgt_lang $pivot2tgt_ckpt_dir
    fi

    # compute automatic string-based metrics if the prediction exists for the system
    if [[ -f "${tgt_fname}.pred.${system}" ]]; then
        echo "Computing Metrics"
        bash compute_metrics.sh $tgt_fname.pred.$system $tgt_fname $tgt_lang > $devtest_data_dir/$src_lang-$tgt_lang/${src_lang}_${tgt_lang}_${system}_scores.txt
    fi

    # remove the intermediate files
    rm $pivot_fname.pred.itv2.*
    rm $tgt_fname.pred.itv2.*
    rm -rf $devtest_data_dir/$src_lang-$tgt_lang/*.tok

done
