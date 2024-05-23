import os
import sys
import regex as re
from tqdm import tqdm
from typing import Iterator, Tuple


def add_tags(
    sent: str,
    src_lang: str,
    tgt_lang: str,
    delimiter: str = " ",
    add_bt_tag: bool = False,
    add_ft_tag: bool = False,
):
    """
    Adds the source and target language tags to the input sentence.

    Args:
        sent (str): input sentence.
        src_lang (str): source language tag.
        tgt_lang (str): target language tag.
        delimiter (str): delimiter to separate the language tags from the sentence.
        add_bt_tag (bool): whether to add the back-translation tag.
        add_ft_tag (bool): whether to add the forward-translation tag.
    Returns:
        str: sentence with the source and target language tags.
    """
    sent = f"{src_lang}{delimiter}{tgt_lang}{delimiter}{sent}"

    assert not (add_bt_tag and add_ft_tag), "Both BT and FT tags cannot be added."

    if add_bt_tag:
        sent = f"__bt__{delimiter}{sent}"
    if add_ft_tag:
        sent = f"__ft__{delimiter}{sent}"

    return sent


def generate_lang_tag_iterator(infname: str) -> Iterator[Tuple[str, str]]:
    """
    Creates an iterator that reads the meta data from `infname` file and
    yields the language tags in the form of tuples "`(src_lang, tgt_lang)`."

    Args:
        infname (str): path of the input filename from which the metadata will be read.

    Yields:
        Iterator[Tuple[str, str]]: an iterator that yields source and target language tags
        in the form of tuples.
    """
    with open(infname, "r", encoding="utf-8") as infile:
        for line in infile:
            src_lang, tgt_lang, count = line.strip().split("\t")
            count = int(count)
            for _ in range(count):
                yield (src_lang, tgt_lang)


if __name__ == "__main__":

    expdir = sys.argv[1]
    split = sys.argv[2]
    add_bt_tag = sys.argv[3]
    add_ft_tag = sys.argv[4]

    add_bt_tag = add_bt_tag.lower() == "true" and split == "train"
    add_ft_tag = add_ft_tag.lower() == "true" and split == "train"

    src_fname = os.path.join(expdir, "bpe", f"{split}.SRC")
    tgt_fname = os.path.join(expdir, "bpe", f"{split}.TGT")
    meta_fname = os.path.join(expdir, "data", f"{split}_lang_pairs.txt")

    out_src_fname = os.path.join(expdir, "final", f"{split}.SRC")
    out_tgt_fname = os.path.join(expdir, "final", f"{split}.TGT")

    lang_tag_iterator = generate_lang_tag_iterator(meta_fname)

    os.makedirs(os.path.join(expdir, "final"), exist_ok=True)

    with open(src_fname, "r", encoding="utf-8") as src_file, open(
        tgt_fname, "r", encoding="utf-8"
    ) as tgt_file, open(out_src_fname, "w", encoding="utf-8") as out_src_file, open(
        out_tgt_fname, "w", encoding="utf-8"
    ) as out_tgt_file:

        for (src_lang, tgt_lang), src_sent, tgt_sent in tqdm(
            zip(lang_tag_iterator, src_file, tgt_file)
        ):
            out_src_file.write(
                add_tags(src_sent.strip(), src_lang, tgt_lang, add_bt_tag=add_bt_tag)
                + "\n"
            )
            out_tgt_file.write(tgt_sent.strip() + "\n")
