"""
FLORES language code mapping to 2 letter ISO language code for compatibility 
with Indic NLP Library (https://github.com/anoopkunchukuttan/indic_nlp_library)
"""
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

iso_to_flores = {iso_code: flores_code for flores_code, iso_code in flores_to_iso.items()}
# Patch for digraphic langs.
iso_to_flores["ks"] = "kas_Arab"
iso_to_flores["ks_Deva"] = "kas_Deva"
iso_to_flores["mni"] = "mni_Mtei"
iso_to_flores["mni_Beng"] = "mni_Beng"
iso_to_flores["sd"] = "snd_Arab"
iso_to_flores["sd_Deva"] = "snd_Deva"
