#/bin/bash

in_dir=$1
out_dir=$2

fairseq-preprocess \
    --source-lang SRC \
    --target-lang TGT \
    --trainpref $in_dir/train \
    --validpref $in_dir/dev \
    --destdir $out_dir \
    --srcdict $out_dir/dict.SRC.txt \
    --tgtdict $out_dir/dict.TGT.txt \
    --thresholdtgt 5 \
    --workers 24