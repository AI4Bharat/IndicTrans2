#!/bin/bash

# This script computes COMET metrics and also performs significance testing on the evaluation set
# where each subdirectory contains En-X pair


echo `date`
devtest_data_dir=$1                         # path to the evaluation directory
model_name=${2-"Unbabel/wmt22-comet-da"}    # name of the model checkpoint

# predefined list of languages supported by COMET
langs=(asm_Beng ben_Beng guj_Gujr hin_Deva kan_Knda mal_Mlym mar_Deva ory_Orya pan_Guru tam_Taml tel_Telu urd_Arab)

# we predefine a set of systems which we consider for evaluation
# feel free to change the below line in case you want to add or remove any system
system=(google azure nllb mbart50 m2m100 it1 it2)


# iterate over the list of predefined languages
for lang in "${langs[@]}"; do

    mkdir -p "$devtest_data_dir/eng_Latn-$lang/comet"

    # --------------------------------------------------------------
    #                   COMET score computation
    # --------------------------------------------------------------

    # iterate over the list of predefined systems
    for sys in "${system[@]}"; do

        echo "${sys}"

        # en - indic direction
        if [ -f "$devtest_data_dir/eng_Latn-$lang/test.$lang.pred.$sys" ]; then
            echo "eng_Latn-${lang}"

            src_fname=$devtest_data_dir/eng_Latn-$lang/test.eng_Latn
            pred_fname=$devtest_data_dir/eng_Latn-$lang/test.$lang.pred.$sys
            ref_fname=$devtest_data_dir/eng_Latn-$lang/test.$lang
            out_fname=$devtest_data_dir/eng_Latn-$lang/comet/eng_Latn_${lang}_${sys}_comet.txt

            # Compute COMET scores using the `comet-score`
            comet-score -s $src_fname -t $pred_fname -r $ref_fname --gpus 1 --model $model_name --quiet --only_system > $out_fname
        fi

        # indic - en direction
        if [ -f "$devtest_data_dir/eng_Latn-$lang/test.eng_Latn.pred.$sys" ]; then
            echo "${lang}-eng_Latn"

            src_fname=$devtest_data_dir/eng_Latn-$lang/test.$lang
            pred_fname=$devtest_data_dir/eng_Latn-$lang/test.eng_Latn.pred.$sys
            ref_fname=$devtest_data_dir/eng_Latn-$lang/test.eng_Latn
            out_fname=$devtest_data_dir/eng_Latn-$lang/comet/${lang}_eng_Latn_${sys}_comet.txt

            # Compute COMET scores using the `comet-score`
            comet-score -s $src_fname -t $pred_fname -r $ref_fname --gpus 1 --model $model_name --quiet --only_system > $out_fname
        fi

    done

    # --------------------------------------------------------------
    #                  COMET significance testing
    # --------------------------------------------------------------
    
    # en - indic direction
    src_fname=$devtest_data_dir/eng_Latn-$lang/test.eng_Latn
    pred_fname=$devtest_data_dir/eng_Latn-$lang/test.$lang.pred.*
    ref_fname=$devtest_data_dir/eng_Latn-$lang/test.$lang
    out_fname=$devtest_data_dir/eng_Latn-$lang/comet/eng_Latn_${lang}_comet_stat.txt

    # Compute COMET significance scores using the `comet-compare`
    comet-compare -s $src_fname -t $pred_fname -r $ref_fname > $out_fname


    # indic-en direction
    src_fname=$devtest_data_dir/eng_Latn-$lang/test.$lang
    pred_fname=$devtest_data_dir/eng_Latn-$lang/test.eng_Latn.pred.*
    ref_fname=$devtest_data_dir/eng_Latn-$lang/test.eng_Latn
    out_fname=$devtest_data_dir/eng_Latn-$lang/comet/${lang}_eng_Latn_comet_stat.txt

    # Compute COMET significance scores using the `comet-compare`
    comet-compare -s $src_fname -t $pred_fname -r $ref_fname > $out_fname

done
