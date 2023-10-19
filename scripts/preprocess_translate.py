import sys
from indicnlp import loader
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate

import re
from typing import Union
from flores_codes_map_indic import flores_codes

loader.load()
en_tok = MosesTokenizer(lang="en")
en_normalizer = MosesPunctNormalizer()

## preferrable use: https://github.com/VarunGumma/indic_nlp_library


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
        remove_tag (bool, optional): whether to remove the do not translate tags (`<dnt>` and `</dnt>`) from the line of text (default: True).

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

    processed_line_matches = [
        processed_line.replace(processed_line_match, raw_match) 
        for (raw_match, processed_line_match) in zip(raw_matches, processed_line_matches)
    ]
    
    if remove_tag:
        processed_line = re.sub("\s+", " ", processed_line.replace("<dnt>", " ")).strip()
        processed_line = re.sub("\s+", " ", processed_line.replace("</dnt>", " ")).strip()
    
    return processed_line
    

def preprocess(
    lang: str, 
    transliterate: bool = False, 
    remove_tag: bool= True
) -> int:
    """
    Preprocess the text in the input file by normalizing, tokenizing and
    script conversation and write the output to a new file.

    Args:
        lang (str): language of the text in the input file.
        transliterate (bool, optional): whether to transliterate the text in input file to devanagari (default: False).
        remove_tag (bool, optional): whether to remove the do not translate tags (`<dnt>` and `</dnt>`) from the text in input file (default: True).

    Returns:
        int: number of sentences in the input file
    """
    iso_lang = flores_codes[lang]
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer(iso_lang) if iso_lang != 'en' else None

    for line in sys.stdin:
        print(preprocess_line(line, normalizer, lang, transliterate, remove_tag))



if __name__ == "__main__":
    lang = sys.argv[1]
    transliterate = sys.argv[2]
    remove_tag = sys.argv[3]
    
    transliterate = transliterate.lower() == "true"
    remove_tag = remove_tag.lower() == "true"

    preprocess(lang, transliterate, remove_tag)
