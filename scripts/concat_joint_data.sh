#!/bin/bash

# get arguments
in_dir=$1
out_dir=$2
split=$3

# make sure the output directory exists
mkdir -p $out_dir

out_src_fname=$out_dir/$split."SRC"
out_tgt_fname=$out_dir/$split."TGT"

echo $out_src_fname
echo $out_tgt_fname

meta_fname=$out_dir/${split}_lang_pairs.txt

# iterate over each directory in the input directory
for pair in $(ls -d $in_dir/*/); do
    # get the source and target language names
    src_lang=$(basename $pair | cut -d '-' -f 1)
    tgt_lang=$(basename $pair | cut -d '-' -f 2)

    echo "src: $src_lang, tgt: $tgt_lang"

    in_src_fname=$pair/$split.$src_lang
    in_tgt_fname=$pair/$split.$tgt_lang

    # only proceed if both the source and target files exist
    if [[ -f $in_src_fname && -f $in_tgt_fname ]]; then
        echo $in_src_fname
        echo $in_tgt_fname

        # append the contents of the source and target files to the output files
        cat $in_src_fname >> $out_src_fname
        cat $in_tgt_fname >> $out_tgt_fname

        # count the lines in the source file and write to the meta file
        corpus_size=$(grep -c '.' $in_src_fname)
        echo -e "$src_lang\t$tgt_lang\t$corpus_size" >> $meta_fname
    fi
done
