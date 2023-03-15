import sys
from tqdm import tqdm


def add_token(sent: str, src_lang: str, tgt_lang: str, delimiter: str = " ") -> str:
    """
    Add special tokens indicating source and target language to the start of the input sentence.
    The resulting string will have the format: "`{src_lang} {tgt_lang} {input_sentence}`".

    Args:
        sent (str): input sentence to be translated.
        src_lang (str): language of the input sentence.
        tgt_lang (str): language in which the input sentence will be translated.
        delimiter (str): separator to add between language tags and input sentence (default: " ").

    Returns:
        str: input sentence with the special tokens added to the start.
    """
    return src_lang + delimiter + tgt_lang + delimiter + sent


if __name__ == "__main__":
    infname = sys.argv[1]
    outfname = sys.argv[2]
    src_lang = sys.argv[3]
    tgt_lang = sys.argv[4]

    with open(infname, "r", encoding="utf-8") as infile, open(
        outfname, "w", encoding="utf-8"
    ) as outfile:
        for line in tqdm(infile):
            outstr = add_token(line.strip(), src_lang, tgt_lang)
            outfile.write(outstr + "\n")
