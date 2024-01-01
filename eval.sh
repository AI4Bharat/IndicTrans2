#!/bin/bash

echo `date`
devtest_data_dir=$1
ckpt_dir=$2
model=${3:-"base18L"}
system=${2:-"it2"}

pairs=$(ls -rd $devtest_data_dir/*)

echo "Removing old scores"
rm $devtest_data_dir/*.csv

for pair in ${pairs[@]}; do
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)

    src_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$src_lang
    tgt_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang

    if [ -f "$src_fname" ] && [ -f "$tgt_fname" ]; then
        echo "Evaluating $src_lang-$tgt_lang ..."
    else 
        echo "Skipping $src_lang-$tgt_lang ..."
        continue
    fi

    echo "Generating Translations"
    bash joint_translate.sh $src_fname $tgt_fname.pred.$system $src_lang $tgt_lang $ckpt_dir $model

    if [[ -f "${tgt_fname}.pred.${system}" ]]; then
        echo "Computing Metrics"
        bash compute_metrics.sh $tgt_fname.pred.$system $tgt_fname $tgt_lang > $devtest_data_dir/$src_lang-$tgt_lang/${src_lang}_${tgt_lang}_${system}_scores.json
    fi
done

echo "Collating Metrics for ${system}"
python scripts/collate_multi_system_metrics.py --devtest_dir $devtest_data_dir --systems $system