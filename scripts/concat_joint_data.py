import os
import sys
from tqdm import tqdm
from typing import List


def concat_data(
    data_dir: str,
    out_dir: str,
    lang_pair_list: List[List[str]],
    out_src_lang: str = "SRC",
    out_tgt_lang: str = "TGT",
    split: str = "train",
):
    """
    Concatenate data files from different language pairs and writes the output to a specified directory.

    Args:
        data_dir (str): path of the directory containing the data files for language pairs.
        out_dir (str): path of the directory where the output files will be saved.
        lang_pair_list (List[List[str]]): a list of language pairs, where each pair is a list of two strings.
        out_src_lang (str, optional): suffix to use for the source language (default: "SRC").
        out_tgt_lang (str, optional): suffix to use for the source language (default: "TGT").
        split (str, optional): name of the split (e.g. "train", "dev", "test") to concatenate (default: "train").
    """
    os.makedirs(out_dir, exist_ok=True)

    out_src_fname = os.path.join(out_dir, f"{split}.{out_src_lang}")
    out_tgt_fname = os.path.join(out_dir, f"{split}.{out_tgt_lang}")

    print()
    print(out_src_fname)
    print(out_tgt_fname)

    # concatenate data for different language pairs
    if os.path.isfile(out_src_fname):
        os.unlink(out_src_fname)
    if os.path.isfile(out_tgt_fname):
        os.unlink(out_tgt_fname)

    for src_lang, tgt_lang in tqdm(lang_pair_list):
        print("src: {}, tgt:{}".format(src_lang, tgt_lang))

        in_src_fname = os.path.join(data_dir, f"{src_lang}-{tgt_lang}", f"{split}.{src_lang}")
        in_tgt_fname = os.path.join(data_dir, f"{src_lang}-{tgt_lang}", f"{split}.{tgt_lang}")

        if not os.path.exists(in_src_fname) or not os.path.exists(in_tgt_fname):
            continue

        print(in_src_fname)
        os.system("cat {} >> {}".format(in_src_fname, out_src_fname))

        print(in_tgt_fname)
        os.system("cat {} >> {}".format(in_tgt_fname, out_tgt_fname))

    corpus_stats(data_dir, out_dir, lang_pair_list, split)


def corpus_stats(data_dir: str, out_dir: str, lang_pair_list: List[List[str]], split: str):
    """
    Computes statistics for the given language pairs in a corpus and
    writes the results to a file in the output directory.

    Args:
        data_dir (str): path of the directory containing the corpus data.
        out_dir (str): path of the directory where the output file should be written.
        lang_pair_list (List[List[str]]): a list of language pairs as lists of strings in the form "`[src_lang, tgt_lang]`".
        split (str): a string indicating the split (e.g. 'train', 'dev', 'test') of the corpus to consider.
    """
    meta_fname = os.path.join(out_dir, f"{split}_lang_pairs.txt")
    with open(meta_fname, "w", encoding="utf-8") as lp_file:

        for src_lang, tgt_lang in tqdm(lang_pair_list):
            print("src: {}, tgt:{}".format(src_lang, tgt_lang))

            in_src_fname = os.path.join(data_dir, f"{src_lang}-{tgt_lang}", f"{split}.{src_lang}")
            if not os.path.exists(in_src_fname):
                continue

            print(in_src_fname)

            corpus_size = 0
            with open(in_src_fname, "r", encoding="utf-8") as infile:
                corpus_size = sum(map(lambda x: 1, infile))

            lp_file.write(f"{src_lang}\t{tgt_lang}\t{corpus_size}\n")


if __name__ == "__main__":

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    split = sys.argv[3]
    lang_pair_list = []

    pairs = os.listdir(in_dir)
    for pair in pairs:
        src_lang, tgt_lang = pair.split("-")
        lang_pair_list.append([src_lang, tgt_lang])

    concat_data(in_dir, out_dir, lang_pair_list, split=split)
