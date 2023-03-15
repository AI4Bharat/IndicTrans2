INDIC_NLP_LIB_HOME = "indic_nlp_library"
INDIC_NLP_RESOURCES = "indic_nlp_resources"
import sys

sys.path.append(r"{}".format(INDIC_NLP_LIB_HOME))
from indicnlp import common

common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader

loader.load()
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer
from sacremoses import MosesDetokenizer
from collections import defaultdict

from tqdm import tqdm
from joblib import Parallel, delayed

from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_detokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate

import re
from typing import Union
from flores_codes_map_indic import flores_codes

en_tok = MosesTokenizer(lang="en")
en_normalizer = MosesPunctNormalizer()


def preprocess_line(
    line: str,
    normalizer: Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory],
    lang: str,
    transliterate: bool = False,
    remove_tag: bool = True
) -> str:
    """
    Preprocess a line of text by normalizing, tokenization, and possibly transliterating it.

    Args:
        line (str): the line of text to preprocess.
        normalizer (Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory]): an object that performs normalization on the text.
        lang (str): the language of the line of text
        transliterate (bool, optional): whether to transliterate the line of text to devanagari (default: False).

    Returns:
        str: preprocessed line of text.
    """
    iso_lang = flores_codes[lang]
    
    pattern = r'<dnt>(.*?)</dnt>'
    raw_matches = re.findall(pattern, line)

    if iso_lang == "en":
        processed_line = " ".join(en_tok.tokenize(en_normalizer.normalize(line.strip()), escape=False))
    elif transliterate:
        # transliterates from the any specific language to devanagari
        # which is why we specify lang2_code as "hi".
        # line = indic_detokenize.trivial_detokenize(line.strip(), lang)
        processed_line = unicode_transliterate.UnicodeIndicTransliterator.transliterate(
            " ".join(indic_tokenize.trivial_tokenize(normalizer.normalize(line.strip()), iso_lang)),
            iso_lang,
            "hi",
        ).replace(" ् ", "्")
    else:
        # we only need to transliterate for joint training
        processed_line = " ".join(
            indic_tokenize.trivial_tokenize(normalizer.normalize(line.strip()), iso_lang)
        )

    processed_line = processed_line.replace("< dnt >", "<dnt>")
    processed_line = processed_line.replace("< / dnt >", "</dnt>")
    
    processed_line_matches = re.findall(pattern, processed_line)
    for raw_match, processed_line_match in zip(raw_matches, processed_line_matches):
        processed_line = processed_line.replace(processed_line_match, raw_match)
    
    if remove_tag:
        processed_line = re.sub("\s+", " ", processed_line.replace("<dnt>", " ")).strip()
        processed_line = re.sub("\s+", " ", processed_line.replace("</dnt>", " ")).strip()
    
    return processed_line
    

def preprocess(infname: str, outfname: str, lang: str, transliterate: bool = False, remove_tag: bool= True) -> int:
    """
    Preprocess the text in the input file by normalizing, tokenizing and
    script conversation and write the output to a new file.

    Args:
        infname (str): path of the input file.
        outfname (str): path of the output file.
        lang (str): language of the text in the input file.
        transliterate (bool, optional): whether to transliterate the text in input file to devanagari (default: False).

    Returns:
        int: number of sentences in the input file
    """
    iso_lang = flores_codes[lang]

    n = 0
    num_lines = sum(1 for line in open(infname, "r"))

    if iso_lang == "en":
        with open(infname, "r", encoding="utf-8") as infile, open(
            outfname, "w", encoding="utf-8"
        ) as outfile:

            out_lines = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(preprocess_line)(line, None, lang, transliterate, remove_tag) for line in tqdm(infile, total=num_lines)
            )

            for line in out_lines:
                outfile.write(line + "\n")
                n += 1
    else:
        normfactory = indic_normalize.IndicNormalizerFactory()
        normalizer = normfactory.get_normalizer(iso_lang)
        # reading
        with open(infname, "r", encoding="utf-8") as infile, open(
            outfname, "w", encoding="utf-8"
        ) as outfile:

            out_lines = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(preprocess_line)(line, normalizer, lang, transliterate, remove_tag)
                for line in tqdm(infile, total=num_lines)
            )

            for line in out_lines:
                outfile.write(line + "\n")
                n += 1

    return n


if __name__ == "__main__":
    infname = sys.argv[1]
    outfname = sys.argv[2]
    lang = sys.argv[3]
    transliterate = sys.argv[4]
    remove_tag = sys.argv[5]
    
    if transliterate.lower() == "true":
        transliterate = True
    else:
        transliterate = False
    
    if remove_tag.lower() == "true":
        remove_tag = True
    else:
        remove_tag = False

    print(preprocess(infname, outfname, lang, transliterate, remove_tag))
