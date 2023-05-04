#!/bin/bash

echo `date`
devtest_data_dir=$1
ckpt_dir=$2
system=${3:-"itv2"}

pairs=$(ls -d $devtest_data_dir/* | sort)

for pair in ${pairs[@]}; do
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)

    src_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang
    tgt_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$src_lang

    if [ -f "$src_fname" ] && [ -f "$tgt_fname" ]; then
        echo "Evaluating $src_lang-$tgt_lang ..."
    else 
        echo "Skipping $src_lang-$tgt_lang ..."
        continue
    fi

    if [[ $system == *"itv2"* ]]; then
        echo "Generating Translations"
        bash joint_translate.sh $src_fname $tgt_fname.pred.$system $tgt_lang $src_lang $ckpt_dir
    fi

    if [[ -f "${tgt_fname}.pred.${system}" ]]; then
        echo "Computing Metrics"
        bash compute_metrics.sh $tgt_fname.pred.$system $tgt_fname $src_lang > $devtest_data_dir/$src_lang-$tgt_lang/${tgt_lang}_${src_lang}_${system}_scores.txt
    fi

    # Purge the intermediate files to declutter the directory.
    rm -rf $tgt_fname.pred.$system.*
    rm -rf $devtest_data_dir/$src_lang-$tgt_lang/*.tok

done
