#!/bin/bash

in_dir=$1

rm ${in_dir}/*.csv

for pair in `ls $in_dir`; do 
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)
    echo "renaming $pair to ${tgt_lang}-${src_lang}"
    mv $in_dir/$pair $in_dir/${tgt_lang}-${src_lang}
done 