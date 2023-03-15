INDIC_NLP_LIB_HOME = "indic_nlp_library"
INDIC_NLP_RESOURCES = "indic_nlp_resources"
import sys

from indicnlp import transliterate

sys.path.append(r"{}".format(INDIC_NLP_LIB_HOME))
from indicnlp import common

common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader

loader.load()
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer
from sacremoses import MosesDetokenizer
from collections import defaultdict

import indicnlp
from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_detokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate

from flores_codes_map_indic import flores_codes
import sentencepiece as spm

import re

en_detok = MosesDetokenizer(lang="en")


def postprocess(
    infname: str,
    outfname: str,
    input_size: int,
    lang: str,
    transliterate: bool = False,
    spm_model_path: str = None,
):
    """
    Postprocess the output of a machine translation model in the following order:
        - parse fairseq interactive output
        - convert script back to native Indic script (in case of Indic languages)
        - detokenize

    Args:
        infname (str): path to the input file containing the machine translation output.
        outfname (str): path to the output file where the postprocessed output will be written.
        input_size (int): number of sentences in the input file.
        lang (str): language code of the output language.
        transliterate (bool, optional): whether to transliterate the output text to devanagari (default: False).
    """
    if spm_model_path is None:
        raise Exception("Please provide sentence piece model path for decoding")
    
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    
    iso_lang = flores_codes[lang]

    consolidated_testoutput = []
    consolidated_testoutput = [(x, 0.0, "") for x in range(input_size)]

    temp_testoutput = []
    with open(infname, "r", encoding="utf-8") as infile:
        temp_testoutput = list(
            map(
                lambda x: x.strip().split("\t"),
                filter(lambda x: x.startswith("H-"), infile),
            )
        )
        temp_testoutput = list(
            map(lambda x: (int(x[0].split("-")[1]), float(x[1]), x[2]), temp_testoutput)
        )
        for sid, score, hyp in temp_testoutput:
            consolidated_testoutput[sid] = (sid, score, hyp)
        consolidated_testoutput = [x[2] for x in consolidated_testoutput]
        consolidated_testoutput = [sp.decode(x.split(" ")) for x in consolidated_testoutput]

    if iso_lang == "en":
        with open(outfname, "w", encoding="utf-8") as outfile:
            for sent in consolidated_testoutput:
                outfile.write(en_detok.detokenize(sent.split(" ")) + "\n")
    else:
        xliterator = unicode_transliterate.UnicodeIndicTransliterator()
        with open(outfname, "w", encoding="utf-8") as outfile:
            for sent in consolidated_testoutput:
                if transliterate:
                    outstr = indic_detokenize.trivial_detokenize(
                        xliterator.transliterate(sent, "hi", iso_lang), iso_lang
                    )
                else:
                    outstr = indic_detokenize.trivial_detokenize(sent, iso_lang)
                outfile.write(outstr + "\n")


if __name__ == "__main__":
    infname = sys.argv[1]
    outfname = sys.argv[2]
    input_size = int(sys.argv[3])
    lang = sys.argv[4]
    transliterate = sys.argv[5]
    spm_model_path = sys.argv[6]

    postprocess(infname, outfname, input_size, lang, transliterate, spm_model_path)
