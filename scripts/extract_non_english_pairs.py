import os
import sys
from tqdm import tqdm
from typing import Iterator, List, Tuple
from remove_train_devtest_overlaps import pair_dedup_files


def read_file(fname: str) -> Iterator[str]:
    """
    Reads text from the input file and yields the text line-by-line as string.

    Args:
        fname (str): name of the input file to read.

    Yields:
        Iterator[str]: yields text line-by-line as a string from the input file.
    """
    with open(fname, "r", encoding="utf-8") as infile:
        for line in infile:
            yield line.strip()


def extract_non_english_pairs(in_dir: str, out_dir: str, pivot_lang: str, langs: List[str]):
    """
    Extracts non-English language pairs from a parallel corpora using pivot-translation.

    Args:
        in_dir (str): path of the directory where the input files are stored.
        out_dir (str): path of the directory where the output files are stored.
        pivot_lang (str): pivot language that the input files are translated to.
        langs (List[str]): a list of language codes for the non-English languages.
    """
    for i in tqdm(range(len(langs) - 1)):
        print()
        for j in range(i + 1, len(langs)):
            lang1 = langs[i]
            lang2 = langs[j]

            print("{} {}".format(lang1, lang2))

            fname1 = "{}/{}-{}/train.{}".format(in_dir, pivot_lang, lang1, pivot_lang)
            fname2 = "{}/{}-{}/train.{}".format(in_dir, pivot_lang, lang2, pivot_lang)

            enset_l1 = set(read_file(fname1))
            common_en_set = enset_l1.intersection(read_file(fname2))

            il_fname1 = "{}/{}-{}/train.{}".format(in_dir, pivot_lang, lang1, lang1)
            en_lang1_dict = {}
            for en_line, il_line in zip(read_file(fname1), read_file(il_fname1)):
                if en_line in common_en_set:
                    en_lang1_dict[en_line] = il_line

            os.makedirs("{}/{}-{}".format(out_dir, lang1, lang2), exist_ok=True)
            out_l1_fname = "{o}/{l1}-{l2}/train.{l1}".format(o=out_dir, l1=lang1, l2=lang2)
            out_l2_fname = "{o}/{l1}-{l2}/train.{l2}".format(o=out_dir, l1=lang1, l2=lang2)

            il_fname2 = "{}/en-{}/train.{}".format(in_dir, lang2, lang2)
            with open(out_l1_fname, "w", encoding="utf-8") as out_l1_file, open(
                out_l2_fname, "w", encoding="utf-8"
            ) as out_l2_file:
                for en_line, il_line in zip(read_file(fname2), read_file(il_fname2)):
                    if en_line in en_lang1_dict:
                        # this block should be used if you want to consider multiple tranlations.
                        for il_line_lang1 in en_lang1_dict[en_line]:
                            # lang1_line, lang2_line = il_line_lang1, il_line
                            # out_l1_file.write(lang1_line + "\n")
                            # out_l2_file.write(lang2_line + "\n")

                            # this block should be used if you DONT to consider multiple translation.
                            lang1_line, lang2_line = en_lang1_dict[en_line], il_line
                            out_l1_file.write(lang1_line + "\n")
                            out_l2_file.write(lang2_line + "\n")

            pair_dedup_files(out_l1_fname, out_l2_fname)


def get_extracted_stats(out_dir: str, langs: List[str]) -> List[Tuple[str, str, int]]:
    """
    Gathers stats from the extracted non-english pairs.

    Args:
        out_dir (str): path of the directory where the output files are stored.
        langs (List[str]): a list of language codes.

    Returns:
        List[Tuple[str, str, int]]: a list of tuples, where each tuple contains statistical information
            about a language pair in the form "`(lang1, lang2, count)`".
    """
    common_stats = []
    for i in tqdm(range(len(langs) - 1)):
        for j in range(i + 1, len(langs)):
            lang1 = langs[i]
            lang2 = langs[j]

            out_l1_fname = "{o}/{l1}-{l2}/train.{l1}".format(o=out_dir, l1=lang1, l2=lang2)

            cnt = sum([1 for _ in read_file(out_l1_fname)])
            common_stats.append((lang1, lang2, cnt))
            common_stats.append((lang2, lang1, cnt))
        return common_stats


if __name__ == "__main__":
    #TODO: need to fix this

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    langs = sorted([lang.strip() for lang in sys.argv[3].split(",")])

    if len(sys.argv) == 4:
        pivot_lang = "eng_Latn"
    else:
        pivot_lang = sys.argv[4]

    for pair in os.listdir(in_dir):
        src_lang, tgt_lang = pair.split("-")
        if src_lang == pivot_lang:
            continue
        else:
            tmp_in_dir = os.path.join(in_dir, pair)
            tmp_out_dir = os.path.join(in_dir, "{}-{}".format(pivot_lang, src_lang))
            os.rename(tmp_in_dir, tmp_out_dir)

    #extract_non_english_pairs(in_dir, out_dir, pivot_lang, langs)

    """stats = get_extracted_stats(out_dir, langs)
    with open("{}/lang_pairs.txt", "w") as f:
        for stat in stats:
            stat = list(map(str, stat))
            f.write("\t".join(stat) + "\n")
"""