#/bin/bash

exp_dir=$1
vocab_dir=$2
train_data_dir=$3
devtest_data_dir=$4
benchmarks_dir=$5

root=$(dirname $0)

echo "root: ${root}"
echo "Running experiment ${exp_dir}"

rm -rf $exp_dir
rm -rf ${train_data_dir}_benchmarks_deduped

echo "deduping train set wrt benchmarks"
python scripts/dedup_benchmark.py $train_data_dir ${train_data_dir}_benchmarks_deduped $benchmarks_dir

train_data_dir=${train_data_dir}_benchmarks_deduped
echo "train_data_dir set to ${train_data_dir}"

# bash temp_scripts/flip_names.sh $train_data_dir
# bash temp_scripts/flip_names.sh $devtest_data_dir

train_processed_dir=$exp_dir/data
devtest_processed_dir=$exp_dir/data

mkdir -p $train_processed_dir
mkdir -p $devtest_processed_dir

mkdir -p $exp_dir/bpe
mkdir -p $exp_dir/final
mkdir -p $exp_dir/final_bin 
cp -r $vocab_dir/vocab $exp_dir
cp -r $vocab_dir/final_bin/dict.* $exp_dir/final_bin


pairs=$(ls -rd $train_data_dir/*)

for pair in ${pairs[@]}; do
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)

    echo "lang_pair: $pair"

    train_norm_dir=$exp_dir/norm/$pair
	devtest_norm_dir=$exp_dir/norm/$pair
	mkdir -p $train_norm_dir
	mkdir -p $devtest_norm_dir

    src_transliterate="true"
    if [[ $src_lang == *"Arab"* ]] || [[ $src_lang == *"Olck"* ]] || \
        [[ $src_lang == *"Mtei"* ]] || [[ $src_lang == *"Latn"* ]]; then
        src_transliterate="false"
    fi
    
    tgt_transliterate="true"
    if [[ $tgt_lang == *"Arab"* ]] || [[ $tgt_lang == *"Olck"* ]] || \
        [[ $tgt_lang == *"Mtei"* ]] || [[ $tgt_lang == *"Latn"* ]]; then
        tgt_transliterate="false"
    fi

    # --------------------------------------------------------------------------
    # train preprocessing
    # --------------------------------------------------------------------------
	train_infname_src=$train_data_dir/$pair/train.$src_lang
	train_infname_tgt=$train_data_dir/$pair/train.$tgt_lang
	train_outfname_src=$train_norm_dir/train.$src_lang
	train_outfname_tgt=$train_norm_dir/train.$tgt_lang

    # normalize punctuations
    echo "Normalizing punctuations for train"
    parallel --pipe --keep-order bash scripts/normalize_punctuation.sh $src_lang < $train_infname_src > $train_outfname_src._norm
    parallel --pipe --keep-order bash scripts/normalize_punctuation.sh $tgt_lang < $train_infname_tgt > $train_outfname_tgt._norm

	# add do not translate tags to handle special failure cases
    # echo "Applying do not translate tags for train"
    echo "normalizing regex for train"
    python scripts/normalize_regex.py $train_outfname_src._norm $train_outfname_tgt._norm $train_outfname_src.norm $train_outfname_tgt.norm

	echo "Applying normalization and script conversion for train"
	# this is for preprocessing text and in for indic langs, we convert all scripts to devnagiri
    parallel --pipe --keep-order python scripts/preprocess_translate.py $src_lang $src_transliterate false < $train_outfname_src.norm > $train_outfname_src
    parallel --pipe --keep-order python scripts/preprocess_translate.py $tgt_lang $tgt_transliterate true < $train_outfname_tgt.norm > $train_outfname_tgt
    echo "Input size: $(grep -c '.' $train_outfname_src)"

    # --------------------------------------------------------------------------
    # dev preprocessing
    # --------------------------------------------------------------------------
	dev_infname_src=$devtest_data_dir/$pair/dev.$src_lang
	dev_infname_tgt=$devtest_data_dir/$pair/dev.$tgt_lang
	dev_outfname_src=$devtest_norm_dir/dev.$src_lang
	dev_outfname_tgt=$devtest_norm_dir/dev.$tgt_lang

    # normalize punctuations
    echo "Normalizing punctuations for dev"
    parallel --pipe --keep-order bash scripts/normalize_punctuation.sh $src_lang < $dev_infname_src > $dev_outfname_src._norm
    parallel --pipe --keep-order bash scripts/normalize_punctuation.sh $tgt_lang < $dev_infname_tgt > $dev_outfname_tgt._norm

	# add do not translate tags to handle special failure cases
    # echo "Applying do not translate tags for dev"
    echo "normalizing regex for dev"
    python scripts/normalize_regex.py $dev_outfname_src._norm $dev_outfname_tgt._norm $dev_outfname_src.norm $dev_outfname_tgt.norm

    echo "Applying normalization and script conversion for dev"
    parallel --pipe --keep-order python scripts/preprocess_translate.py $src_lang $src_transliterate false < $dev_outfname_src.norm > $dev_outfname_src
    parallel --pipe --keep-order python scripts/preprocess_translate.py $tgt_lang $tgt_transliterate true < $dev_outfname_tgt.norm > $dev_outfname_tgt 
    echo "Input size: $(grep -c '.' $train_outfname_src)"
done


echo `date`
for split in train dev; do
    # this concatenates lang pair data and creates text files to keep track of number of lines in each lang pair.
    # this is imp as for joint training, we will merge all the lang pairs and the individual lang lines info
    # would be required for adding specific lang tags later.

    # the outputs of these scripts will  be text file like this:
    # <lang1> <lang2> <number of lines>
    # lang1-lang2 n1
    # lang1-lang3 n2
    echo "Concatenating data"
    bash scripts/concat_joint_data.sh $exp_dir/norm $exp_dir/data $split
	
    echo "Applying sentence piece for $split"
    bash scripts/apply_sentence_piece.sh $exp_dir $exp_dir/data $exp_dir/bpe SRC TGT $split

    # # this is only required for joint training
    # we apply language tags to the bpe segmented data
    #
    # if we are translating lang1 to lang2 then <lang1 line> will become __src__ <lang1> __tgt__ <lang2> <lang1 line>
    echo "Adding language tags"
    bash scripts/add_joint_tags_translate.py $exp_dir $split
done

echo `date`
# Binarize the training data for using with fairseq train
echo "Binarizing data"
bash scripts/fairseq_binarize.sh $exp_dir/final $exp_dir/final_bin
  
echo -e "[INFO]\tcleaning unnecessary files from exp dir to save space"
rm -rf $exp_dir/bpe $exp_dir/final $exp_dir/data $exp_dir/norm $train_data_dir

echo -e "[INFO]\tcompleted!"
