import regex as re
from joblib import Parallel, delayed
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Union

from sacremoses import MosesPunctNormalizer
from indicnlp.normalize import indic_normalize
from sacremoses import MosesTokenizer, MosesDetokenizer
from indicnlp.transliterate import unicode_transliterate
from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

en_tok = MosesTokenizer(lang="en")
en_normalizer = MosesPunctNormalizer()
en_detok = MosesDetokenizer(lang="en")
xliterator = unicode_transliterate.UnicodeIndicTransliterator()


flores_codes = {
    "asm_Beng": "as",
    "awa_Deva": "hi",
    "ben_Beng": "bn",
    "bho_Deva": "hi",
    "brx_Deva": "hi",
    "doi_Deva": "hi",
    "eng_Latn": "en",
    "gom_Deva": "kK",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ur",
    "kas_Deva": "hi",
    "kha_Latn": "en",
    "lus_Latn": "en",
    "mag_Deva": "hi",
    "mai_Deva": "hi",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "bn",
    "mni_Mtei": "hi",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "hi",
    "sat_Olck": "or",
    "snd_Arab": "ur",
    "snd_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}


flores_to_iso = {
    "asm_Beng": "as",
    "awa_Deva": "awa",
    "ben_Beng": "bn",
    "bho_Deva": "bho",
    "brx_Deva": "brx",
    "doi_Deva": "doi",
    "eng_Latn": "en",
    "gom_Deva": "gom",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hne",
    "kan_Knda": "kn",
    "kas_Arab": "ksa",
    "kas_Deva": "ksd",
    "kha_Latn": "kha",
    "lus_Latn": "lus",
    "mag_Deva": "mag",
    "mai_Deva": "mai",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "mnib",
    "mni_Mtei": "mnim",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "sa",
    "sat_Olck": "sat",
    "snd_Arab": "sda",
    "snd_Deva": "sdd",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}


INDIC_NUM_MAP = {
    "\u09e6": "0",
    "0": "0",
    "\u0ae6": "0",
    "\u0ce6": "0",
    "\u0966": "0",
    "\u0660": "0",
    "\uabf0": "0",
    "\u0b66": "0",
    "\u0a66": "0",
    "\u1c50": "0",
    "\u06f0": "0",
    "\u09e7": "1",
    "1": "1",
    "\u0ae7": "1",
    "\u0967": "1",
    "\u0ce7": "1",
    "\u06f1": "1",
    "\uabf1": "1",
    "\u0b67": "1",
    "\u0a67": "1",
    "\u1c51": "1",
    "\u0c67": "1",
    "\u09e8": "2",
    "2": "2",
    "\u0ae8": "2",
    "\u0968": "2",
    "\u0ce8": "2",
    "\u06f2": "2",
    "\uabf2": "2",
    "\u0b68": "2",
    "\u0a68": "2",
    "\u1c52": "2",
    "\u0c68": "2",
    "\u09e9": "3",
    "3": "3",
    "\u0ae9": "3",
    "\u0969": "3",
    "\u0ce9": "3",
    "\u06f3": "3",
    "\uabf3": "3",
    "\u0b69": "3",
    "\u0a69": "3",
    "\u1c53": "3",
    "\u0c69": "3",
    "\u09ea": "4",
    "4": "4",
    "\u0aea": "4",
    "\u096a": "4",
    "\u0cea": "4",
    "\u06f4": "4",
    "\uabf4": "4",
    "\u0b6a": "4",
    "\u0a6a": "4",
    "\u1c54": "4",
    "\u0c6a": "4",
    "\u09eb": "5",
    "5": "5",
    "\u0aeb": "5",
    "\u096b": "5",
    "\u0ceb": "5",
    "\u06f5": "5",
    "\uabf5": "5",
    "\u0b6b": "5",
    "\u0a6b": "5",
    "\u1c55": "5",
    "\u0c6b": "5",
    "\u09ec": "6",
    "6": "6",
    "\u0aec": "6",
    "\u096c": "6",
    "\u0cec": "6",
    "\u06f6": "6",
    "\uabf6": "6",
    "\u0b6c": "6",
    "\u0a6c": "6",
    "\u1c56": "6",
    "\u0c6c": "6",
    "\u09ed": "7",
    "7": "7",
    "\u0aed": "7",
    "\u096d": "7",
    "\u0ced": "7",
    "\u06f7": "7",
    "\uabf7": "7",
    "\u0b6d": "7",
    "\u0a6d": "7",
    "\u1c57": "7",
    "\u0c6d": "7",
    "\u09ee": "8",
    "8": "8",
    "\u0aee": "8",
    "\u096e": "8",
    "\u0cee": "8",
    "\u06f8": "8",
    "\uabf8": "8",
    "\u0b6e": "8",
    "\u0a6e": "8",
    "\u1c58": "8",
    "\u0c6e": "8",
    "\u09ef": "9",
    "9": "9",
    "\u0aef": "9",
    "\u096f": "9",
    "\u0cef": "9",
    "\u06f9": "9",
    "\uabf9": "9",
    "\u0b6f": "9",
    "\u0a6f": "9",
    "\u1c59": "9",
    "\u0c6f": "9",
}


multispace_regex = re.compile("[ ]{2,}")
end_bracket_space_punc_regex = re.compile(r"\) ([\.!:?;,])")
digit_space_percent = re.compile(r"(\d) %")
double_quot_punc = re.compile(r"\"([,\.]+)")
digit_nbsp_digit = re.compile(r"(\d) (\d)")


def punc_norm(text, lang="en"):
    text = (
        text.replace("\r", "")
        .replace("(", " (")
        .replace(")", ") ")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace(" :", ":")
        .replace(" ;", ";")
        .replace("`", "'")
        .replace("„", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", " - ")
        .replace("´", "'")
        .replace("‘", "'")
        .replace("‚", "'")
        .replace("’", "'")
        .replace("''", '"')
        .replace("´´", '"')
        .replace("…", "...")
        .replace(" « ", ' "')
        .replace("« ", '"')
        .replace("«", '"')
        .replace(" » ", '" ')
        .replace(" »", '"')
        .replace("»", '"')
        .replace(" %", "%")
        .replace("nº ", "nº ")
        .replace(" :", ":")
        .replace(" ºC", " ºC")
        .replace(" cm", " cm")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ;", ";")
        .replace(", ", ", ")
    )

    text = multispace_regex.sub(" ", text)
    text = end_bracket_space_punc_regex.sub(r")\1", text)
    text = digit_space_percent.sub(r"\1%", text)
    text = double_quot_punc.sub(
        r'\1"', text
    )  # English "quotation," followed by comma, style
    text = digit_nbsp_digit.sub(r"\1.\2", text)  # What does it mean?
    return text.strip(" ")


URL_PATTERN = r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b"
EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"
# handles dates, time, percentages, proportion, ratio, etc
NUMERAL_PATTERN = r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
# handles upi, social media handles and hashtags
OTHER_PATTERN = r"[A-Za-z0-9]*[#|@]\w+"


def normalize_indic_numerals(line: str):
    """
    Normalize the numerals in Indic languages from native script to Roman script (if present).

    Args:
        line (str): an input string with Indic numerals to be normalized.

    Returns:
        str: an input string with the all Indic numerals normalized to Roman script.
    """
    return "".join([INDIC_NUM_MAP.get(c, c) for c in line])


def wrap_with_placeholders(text: str, patterns: list) -> Tuple[str, dict]:
    """
    Wraps substrings with matched patterns in the given text with placeholders and returns
    the modified text along with a mapping of the placeholders to their original value.

    Args:
        text (str): an input string which needs to be wrapped with the placeholders.
        pattern (list): list of patterns to search for in the input string.

    Returns:
        Tuple[str, dict]: a tuple containing the modified text and a dictionary mapping
            placeholders to their original values.
    """
    serial_no = 1

    placeholder_entity_map = dict()

    for pattern in patterns:
        matches = set(re.findall(pattern, text))

        # wrap common match with placeholder tags
        for match in matches:
            if pattern == URL_PATTERN:
                # Avoids false positive URL matches for names with initials.
                temp = match.replace(".", "")
                if len(temp) < 4:
                    continue
            if pattern == NUMERAL_PATTERN:
                # Short numeral patterns do not need placeholder based handling.
                temp = match.replace(" ", "").replace(".", "").replace(":", "")
                if len(temp) < 4:
                    continue

            # Set of Translations of "ID" in all the suppported languages have been collated.
            # This has been added to deal with edge cases where placeholders might get translated.
            indic_failure_cases = [
                "آی ڈی ",
                "ꯑꯥꯏꯗꯤ",
                "आईडी",
                "आई . डी . ",
                "आई . डी .",
                "आई. डी. ",
                "आई. डी.",
                "ऐटि",
                "آئی ڈی ",
                "ᱟᱭᱰᱤ ᱾",
                "आयडी",
                "ऐडि",
                "आइडि",
            ]
            placeholder = "<ID{}>".format(serial_no)
            alternate_placeholder = "< ID{} >".format(serial_no)
            placeholder_entity_map[placeholder] = match
            placeholder_entity_map[alternate_placeholder] = match
            placeholder = "<ID{}]".format(serial_no)
            alternate_placeholder = "< ID{} ]".format(serial_no)
            placeholder_entity_map[placeholder] = match
            placeholder_entity_map[alternate_placeholder] = match

            for i in indic_failure_cases:
                placeholder_temp = "<{}{}>".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "< {}{} >".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "< {} {} >".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "<{} {}]".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "< {} {} ]".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "[{} {}]".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match
                placeholder_temp = "[ {} {} ]".format(i, serial_no)
                placeholder_entity_map[placeholder_temp] = match

            text = text.replace(match, placeholder)
            serial_no += 1

    text = re.sub("\s+", " ", text)

    # Regex has failure cases in trailing "/" in URLs, so this is a workaround.
    text = text.replace(">/", ">")
    text = text.replace("]/", "]")

    return text, placeholder_entity_map


def normalize(
    text: str,
    patterns: list = [EMAIL_PATTERN, URL_PATTERN, NUMERAL_PATTERN, OTHER_PATTERN],
) -> Tuple[str, dict]:
    """
    Normalizes and wraps the spans of input string with placeholder tags. It first normalizes
    the Indic numerals in the input string to Roman script. Later, it uses the input string with normalized
    Indic numerals to wrap the spans of text matching the pattern with placeholder tags.

    Args:
        text (str): input string.
        pattern (list): list of patterns to search for in the input string.

    Returns:
        Tuple[str, dict]: a tuple containing the modified text and a dictionary mapping
            placeholders to their original values.
    """
    text = normalize_indic_numerals(text.strip("\n"))
    text, placeholder_entity_map = wrap_with_placeholders(text, patterns)
    return text, placeholder_entity_map


def split_sentences(paragraph: str, lang: str) -> List[str]:
    """
    Splits the input text paragraph into sentences. It uses `moses` for English and
    `indic-nlp` for Indic languages.

    Args:
        paragraph (str): input text paragraph.
        lang (str): flores language code.

    Returns:
        List[str] -> list of sentences.
    """
    # fails to handle sentence splitting in case of
    # with MosesSentenceSplitter(lang) as splitter:
    #     return splitter([paragraph])
    return (
        sent_tokenize(paragraph)
        if lang == "eng_Latn"
        else sentence_split(
            paragraph, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA
        )
    )


def apply_lang_tags(sents: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    """
    Add special tokens indicating source and target language to the start of the each input sentence.
    Each resulting input sentence will have the format: "`{src_lang} {tgt_lang} {input_sentence}`".

    Args:
        sent (str): input sentence to be translated.
        src_lang (str): flores lang code of the input sentence.
        tgt_lang (str): flores lang code in which the input sentence will be translated.

    Returns:
        List[str]: list of input sentences with the special tokens added to the start.
    """
    return Parallel(n_jobs=-1)(
        delayed(lambda x: f"{src_lang} {tgt_lang} {x.strip()}")(sent) for sent in sents
    )


def preprocess_sent(
    sent: str,
    normalizer: Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory],
    lang: str,
) -> str:
    """
    Preprocess an input text sentence by normalizing, tokenization, and possibly transliterating it.

    Args:
        sent (str): input text sentence to preprocess.
        normalizer (Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory]): an object that performs normalization on the text.
        lang (str): flores language code of the input text sentence.

    Returns:
        Tuple[str, dict]: a tuple of preprocessed input text sentence and also a corresponding dictionary
            mapping placeholders to their original values.
    """
    iso_lang = flores_codes[lang]
    sent = punc_norm(sent, iso_lang)
    sent, placeholder_entity_map = normalize(sent)

    transliterate = True
    if lang.split("_")[1] in ["Arab", "Aran", "Olck", "Mtei", "Latn"]:
        transliterate = False

    if iso_lang == "en":
        processed_sent = " ".join(
            en_tok.tokenize(en_normalizer.normalize(sent.strip()), escape=False)
        )
    elif transliterate:
        # transliterates from the any specific language to devanagari
        # which is why we specify lang2_code as "hi".
        processed_sent = xliterator.transliterate(
            " ".join(
                indic_tokenize.trivial_tokenize(
                    normalizer.normalize(sent.strip()), iso_lang
                )
            ),
            iso_lang,
            "hi",
        ).replace(" ् ", "्")
    else:
        # we only need to transliterate for joint training
        processed_sent = " ".join(
            indic_tokenize.trivial_tokenize(
                normalizer.normalize(sent.strip()), iso_lang
            )
        )

    return processed_sent, placeholder_entity_map


def preprocess(sents: List[str], lang: str):
    """
    Preprocess an array of sentences by normalizing, tokenization, and possibly transliterating it.

    Args:
        batch (List[str]): input list of sentences to preprocess.
        lang (str): flores language code of the input text sentences.

    Returns:
        Tuple[List[str], List[dict]]: a tuple of list of preprocessed input text sentences and also a corresponding list of dictionary
            mapping placeholders to their original values.
    """

    normalizer = (
        indic_normalize.IndicNormalizerFactory().get_normalizer(flores_codes[lang])
        if lang != "eng_Latn"
        else None
    )

    processed_sents, placeholder_entity_map_sents = zip(
        *[preprocess_sent(sent, normalizer, lang) for sent in sents]
    )

    return processed_sents, placeholder_entity_map_sents


def preprocess_batch(batch: List[str], src_lang: str, tgt_lang: str, is_target: bool = False) -> List[str]:
    """
    Preprocess an array of sentences by normalizing, tokenization, and possibly transliterating it. It also tokenizes the
    normalized text sequences using sentence piece tokenizer and also adds language tags.

    Args:
        batch (List[str]): input list of sentences to preprocess.
        src_lang (str): flores language code of the input text sentences.
        tgt_lang (str): flores language code of the output text sentences.
        is_target (str): add language tags if false otherwise skip it.

    Returns:
        Tuple[List[str], List[dict]]: a tuple of list of preprocessed input text sentences and also a corresponding list of dictionary
            mapping placeholders to their original values.
    """
    preprocessed_sents, placeholder_entity_map_sents = preprocess(batch, lang=src_lang)
    if not is_target:
        tagged_sents = apply_lang_tags(preprocessed_sents, src_lang, tgt_lang)
    else:
        tagged_sents = list(preprocessed_sents)
    return tagged_sents, placeholder_entity_map_sents


def postprocess_batch(
    sents: List[str],
    placeholder_entity_map: List[dict],
    lang: str,
    common_lang: str = "hin_Deva",
) -> List[str]:
    """
    Postprocesses a batch of input sentences after the translation generations.

    Args:
        sents (List[str]): batch of translated sentences to postprocess.
        placeholder_entity_map (List[dict]): dictionary mapping placeholders to the original entity values.
        lang (str): flores language code of the input sentences.
        common_lang (str, optional): flores language code of the transliterated language (defaults: hin_Deva).

    Returns:
        List[str]: postprocessed batch of input sentences.
    """

    lang_code, script_code = lang.split("_")

    for i in range(len(sents)):
        sents[i] = sents[i].replace(" ", "").replace("▁", " ").strip()

        # Fixes for Perso-Arabic scripts
        # TODO: Move these normalizations inside indic-nlp-library
        if script_code in {"Arab", "Aran"}:
            # UrduHack adds space before punctuations. Since the model was trained without fixing this issue, let's fix it now
            sents[i] = sents[i].replace(" ؟", "؟").replace(" ۔", "۔").replace(" ،", "،")
            # Kashmiri bugfix for palatalization: https://github.com/AI4Bharat/IndicTrans2/issues/11
            sents[i] = sents[i].replace("ٮ۪", "ؠ")

        # Oriya bug: indic-nlp-library produces ଯ଼ instead of ୟ when converting from Devanagari to Odia
        # TODO: Find out what's the issue with unicode transliterator for Oriya and fix it
        if lang_code == "or":
            sents[i] = sents[i].replace("ଯ଼", "ୟ")

    assert len(sents) == len(placeholder_entity_map)

    # Replace the placeholders entity
    for i in range(0, len(sents)):
        for key in placeholder_entity_map[i].keys():
            sents[i] = sents[i].replace(key, placeholder_entity_map[i][key])

    # Detokenize and transliterate to native scripts if applicable

    if lang == "eng_Latn":
        postprocessed_sents = [en_detok.detokenize(sent.split(" ")) for sent in sents]
    else:
        postprocessed_sents = [
            indic_detokenize.trivial_detokenize(
                xliterator.transliterate(
                    s, flores_codes[common_lang], flores_codes[lang]
                ),
                flores_codes[lang],
            )
            for s in sents
        ]

    assert len(postprocessed_sents) == len(placeholder_entity_map)

    return postprocessed_sents
