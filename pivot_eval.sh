#!/bin/bash
echo `date`

devtest_data_dir=$1
pivot_lang=${2:-"eng_Latn"}
src2pivot_ckpt_dir=$3
pivot2tgt_ckpt_dir=$4

pairs=$(ls -d $devtest_data_dir/*)

for pair in ${pairs[@]}; do
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)

    src_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$src_lang
    pivot_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$pivot_lang
    tgt_fname=$devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang

    if [[ -f "$devtest_data_dir/$src_lang-$tgt_lang/${src_lang}_${tgt_lang}_itv2_scores.txt" ]]; then
        continue
    fi

    if [ -f "$src_fname" ] && [ -f "$tgt_fname" ]; then
        echo "Evaluating $src_lang-$tgt_lang ..."
    else 
        echo "Skipping $src_lang-$tgt_lang ..."
        continue
    fi

    echo "Generating Source to Pivot Translations"

    # Source to Pivot Translation
    bash joint_translate.sh $src_fname $pivot_fname.pred.itv2 $src_lang $pivot_lang $src2pivot_ckpt_dir
    
    # Purge the intermediate files to declutter the directory.
    rm $pivot_fname.pred.itv2.*

    echo "Generating Pivot to Target Translations"

    # Pivot to Target Translation
    bash joint_translate.sh $pivot_fname.pred.itv2 $tgt_fname.pred.itv2 $pivot_lang $tgt_lang $pivot2tgt_ckpt_dir
    
    # Purge the intermediate files to declutter the directory.
    rm $tgt_fname.pred.itv2.*

    echo "Computing Metrics"
    bash compute_metrics.sh $tgt_fname.pred.itv2 $tgt_fname $tgt_lang > $devtest_data_dir/$src_lang-$tgt_lang/${src_lang}_${tgt_lang}_itv2_scores.txt

done
