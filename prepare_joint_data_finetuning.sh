#/bin/bash

exp_dir=$1
vocab_dir=${2:-"$exp_dir/vocab"}
train_data_dir=${3:-"$exp_dir/train"}
devtest_data_dir=${4:-"$exp_dir/devtest/all"}

root=$(dirname $0)

echo "Running experiment ${exp_dir}"

train_processed_dir=$exp_dir/data
devtest_processed_dir=$exp_dir/data

out_data_dir=$exp_dir/final_bin

mkdir -p $train_processed_dir
mkdir -p $devtest_processed_dir
mkdir -p $out_data_dir

pairs=$(ls -d $train_data_dir/*)

for pair in ${pairs[@]}; do
    pair=$(basename $pair)
    src_lang=$(echo "$pair" | cut -d "-" -f 1)
    tgt_lang=$(echo "$pair" | cut -d "-" -f 2)
    echo "$src_lang - $tgt_lang"

    train_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	devtest_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
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
	train_infname_src=$train_data_dir/${src_lang}-${tgt_lang}/train.$src_lang
	train_infname_tgt=$train_data_dir/${src_lang}-${tgt_lang}/train.$tgt_lang
	train_outfname_src=$train_norm_dir/train.$src_lang
	train_outfname_tgt=$train_norm_dir/train.$tgt_lang

    # normalize punctuations
    echo "Normalizing punctuations for train"
    parallel --pipe --keep-order bash $root/normalize_punctuation.sh $src_lang < $train_infname_src > $train_outfname_src._norm
    parallel --pipe --keep-order bash $root/normalize_punctuation.sh $tgt_lang < $train_infname_tgt > $train_outfname_tgt._norm

	# add do not translate tags to handle special failure cases
    # echo "Applying do not translate tags for train"
    python3 scripts/normalize_regex.py $train_outfname_src._norm $train_outfname_tgt._norm $train_outfname_src.norm $train_outfname_tgt.norm

	echo "Applying normalization and script conversion for train"
	# this is for preprocessing text and in for indic langs, we convert all scripts to devnagiri
	input_size=`python3 scripts/preprocess_translate.py $train_outfname_src.norm $train_outfname_src $src_lang $src_transliterate false`
	input_size=`python3 scripts/preprocess_translate.py $train_outfname_tgt.norm $train_outfname_tgt $tgt_lang $tgt_transliterate true`
	echo "Number of sentences in train: $input_size"

    # --------------------------------------------------------------------------
    # dev preprocessing
    # --------------------------------------------------------------------------
	dev_infname_src=$devtest_data_dir/${src_lang}-${tgt_lang}/dev.$src_lang
	dev_infname_tgt=$devtest_data_dir/${src_lang}-${tgt_lang}/dev.$tgt_lang
	dev_outfname_src=$devtest_norm_dir/dev.$src_lang
	dev_outfname_tgt=$devtest_norm_dir/dev.$tgt_lang

    # normalize punctuations
    echo "Normalizing punctuations for dev"
    parallel --pipe --keep-order bash normalize_punctuation.sh $src_lang < $dev_infname_src > $dev_outfname_src._norm
    parallel --pipe --keep-order bash normalize_punctuation.sh $tgt_lang < $dev_infname_tgt > $dev_outfname_tgt._norm

	# add do not translate tags to handle special failure cases
    # echo "Applying do not translate tags for dev"
    python3 scripts/normalize_regex.py $dev_outfname_src._norm $dev_outfname_tgt._norm $dev_outfname_src.norm $dev_outfname_tgt.norm

    echo "Applying normalization and script conversion for dev"
	input_size=`python scripts/preprocess_translate.py $dev_outfname_src.norm $dev_outfname_src $src_lang $src_transliterate false`
	input_size=`python scripts/preprocess_translate.py $dev_outfname_tgt.norm $dev_outfname_tgt $tgt_lang $tgt_transliterate true`
	echo "Number of sentences in dev: $input_size"

    # # --------------------------------------------------------------------------
    # # test preprocessing
    # # --------------------------------------------------------------------------
	# test_infname_src=$devtest_data_dir/${src_lang}-${tgt_lang}/test.$src_lang
	# test_infname_tgt=$devtest_data_dir/${src_lang}-${tgt_lang}/test.$tgt_lang
	# test_outfname_src=$devtest_norm_dir/test.$src_lang
	# test_outfname_tgt=$devtest_norm_dir/test.$tgt_lang

    # # normalize punctuations
    # echo "Normalizing punctuations for test"
    # parallel --pipe --keep-order bash $root/normalize_punctuation.sh $src_lang < $test_infname_src > $test_outfname_src._norm
    # parallel --pipe --keep-order bash $root/normalize_punctuation.sh $tgt_lang < $test_infname_tgt > $test_outfname_tgt._norm

	# # add do not translate tags to handle special failure cases
    # # echo "Applying do not translate tags for test"
    # python3 scripts/normalize_regex.py $test_outfname_src._norm $test_outfname_tgt._norm $test_outfname_src.norm $test_outfname_tgt.norm

	# echo "Applying normalization and script conversion for test"
	# input_size=`python scripts/preprocess_translate.py $test_outfname_src.norm $test_outfname_src $src_lang $src_transliterate false`
	# input_size=`python scripts/preprocess_translate.py $test_outfname_tgt.norm $test_outfname_tgt $tgt_lang $tgt_transliterate true`
	# echo "Number of sentences in test: $input_size"
done

# this concatenates lang pair data and creates text files to keep track of number of lines in each lang pair.
# this is imp as for joint training, we will merge all the lang pairs and the indivitual lang lines info
# would be required for adding specific lang tags later.

# the outputs of these scripts will  be text file like this:
# <lang1> <lang2> <number of lines>
# lang1-lang2 n1
# lang1-lang3 n2
python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data 'train'
python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data 'dev'
# # python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data 'test'

echo `date`

mkdir -p $exp_dir/bpe

# splits=(train dev test)
splits=(train dev)
for split in ${splits[@]}; do
	echo "Applying sentence piece for $split"

	bash apply_sentence_piece.sh $exp_dir $exp_dir/data $exp_dir/bpe SRC TGT $split

	# # for very large datasets, use gnu-parallel to speed up applying bpe
    # # uncomment the below line if the spm_encode is slow

	# parallel --pipe --keep-order \
	# spm_encode --model=$vocab_dir/vocab/model.SRC \
	# 	--output_format=piece \
	# 	< $exp_dir/data/$split.SRC \
	# 	> $exp_dir/bpe/$split.SRC
	
	# parallel --pipe --keep-order \
	# spm_encode --model=$vocab_dir/vocab/model.TGT \
	# 	--output_format=piece \
	# 	< $exp_dir/data/$split.TGT \
	# 	> $exp_dir/bpe/$split.TGT
done

mkdir -p $exp_dir/final

# # this is only required for joint training
# we apply language tags to the bpe segmented data
#
# if we are translating lang1 to lang2 then <lang1 line> will become __src__ <lang1> __tgt__ <lang2> <lang1 line>
# echo "Adding language tags"
python scripts/add_joint_tags_translate.py $exp_dir 'train'
python scripts/add_joint_tags_translate.py $exp_dir 'dev'
# python scripts/add_joint_tags_translate.py $exp_dir 'test'

# this is important step if you are training with tpu and using num_batch_buckets
# the currnet implementation does not remove outliers before bucketing and hence
# removing these large sentences ourselves helps with getting better buckets
# python scripts/remove_large_sentences.py $exp_dir/bpe/train.SRC $exp_dir/bpe/train.TGT $exp_dir/final/train.SRC $exp_dir/final/train.TGT
# python scripts/remove_large_sentences.py $exp_dir/bpe/dev.SRC $exp_dir/bpe/dev.TGT $exp_dir/final/dev.SRC $exp_dir/final/dev.TGT
# python scripts/remove_large_sentences.py $exp_dir/bpe/test.SRC $exp_dir/bpe/test.TGT $exp_dir/final/test.SRC $exp_dir/final/test.TGT

echo "Binarizing data"
# Binarize the training data for using with fairseq train

# use cpu_count to get num_workers instead of setting it manually when running in different
# instances
num_workers=`python -c "import multiprocessing; print(multiprocessing.cpu_count())"`

data_dir=$exp_dir/final
out_data_dir=$exp_dir/final_bin

# rm -rf $out_data_dir

# add the following line during the release of the repository
#     --testpref $data_dir/test \
fairseq-preprocess \
    --source-lang SRC --target-lang TGT \
    --trainpref $data_dir/train \
    --validpref $data_dir/dev \
    --destdir $out_data_dir \
    --workers $num_workers \
    --srcdict $exp_dir/final_bin/dict.SRC.txt \
    --tgtdict $exp_dir/final_bin/dict.TGT.txt \
