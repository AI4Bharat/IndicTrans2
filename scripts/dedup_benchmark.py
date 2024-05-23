import re
import os
import sys
from collections import defaultdict
from tqdm import tqdm


def remove_overlaps(in_data_dir: str, out_data_dir: str, benchmark_dir: str):
    """
    Removes overlapping sentences between train dataset and dev/test dataset from the
    input directory and writes de-duplicated train data to the specified output directory.

    Args:
        in_data_dir (str): path of the directory containing train data for each language pair.
        out_data_dir (str): path of the directory where the de-duplicated train data will be written for each language pair.
        benchmark_dir (str): path of the directory containing the language-wise monolingual side of dev/test set.
    """
    # load dev/test dataset for each language
    devtest_normalized = defaultdict(set)
    for lang in os.listdir(benchmark_dir):
        fname = os.path.join(benchmark_dir, lang)

        with open(fname, "r") as f:
            sents = [sent for sent in f.read().split("\n") if sent.strip()]
            sents = [
                re.sub(" +", " ", sent).replace("\n", "").strip() for sent in sents
            ]
        devtest_normalized[lang] = set(sents)

    # process each language pair train dataset to remove overlapping sentences
    pairs = sorted(os.listdir(in_data_dir))
    for pair in pairs:
        print(pair)
        src_lang, tgt_lang = pair.split("-")

        src_infname = os.path.join(in_data_dir, pair, f"train.{src_lang}")
        tgt_infname = os.path.join(in_data_dir, pair, f"train.{tgt_lang}")

        src_outfname = os.path.join(out_data_dir, pair, f"train.{src_lang}")
        tgt_outfname = os.path.join(out_data_dir, pair, f"train.{tgt_lang}")

        os.makedirs(os.path.join(out_data_dir, pair), exist_ok=True)

        # remove overlapping sentences and write de-duplicated train data to output directory
        with open(src_infname, "r", encoding="utf-8") as src_infile, open(
            tgt_infname, "r", encoding="utf-8"
        ) as tgt_infile, open(src_outfname, "w", encoding="utf-8") as src_outfile, open(
            tgt_outfname, "w", encoding="utf-8"
        ) as tgt_outfile:

            for src_line, tgt_line in tqdm(zip(src_infile, tgt_infile)):
                src_line = re.sub(" +", " ", src_line).replace("\n", "").strip()
                tgt_line = re.sub(" +", " ", tgt_line).replace("\n", "").strip()

                if (
                    src_line in devtest_normalized[src_lang]
                    or tgt_line in devtest_normalized[tgt_lang]
                ):
                    continue

                src_outfile.write(src_line + "\n")
                tgt_outfile.write(tgt_line + "\n")


if __name__ == "__main__":
    in_data_dir = sys.argv[1]
    out_data_dir = sys.argv[2]
    benchmark_dir = sys.argv[3]

    os.makedirs(out_data_dir, exist_ok=True)

    remove_overlaps(in_data_dir, out_data_dir, benchmark_dir)
