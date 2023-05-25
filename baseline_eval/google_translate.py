import os
import sys
import glob
from tqdm import tqdm
from google.cloud import translate

# Expects a json file containing the API credentials.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(__file__), r"api_key.json"
)

flores_to_iso = {
    "asm_Beng": "as",
    "ben_Beng": "bn",
    "doi_Deva": "doi",
    "eng_Latn": "en",
    "gom_Deva": "gom",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "kan_Knda": "kn",
    "mai_Deva": "mai",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Mtei": "mni_Mtei",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "sa",
    "sat_Olck": "sat",
    "snd_Arab": "sd",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}


# Copy the project id from the json file containing API credentials
def translate_text(text, src_lang, tgt_lang, project_id="project_id"):

    src_lang = flores_to_iso[src_lang]
    tgt_lang = flores_to_iso[tgt_lang]

    if src_lang == "mni_Mtei":
        src_lang = "mni-Mtei"

    if tgt_lang == "mni_Mtei":
        tgt_lang = "mni-Mtei"

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": src_lang,
            "target_language_code": tgt_lang,
        }
    )

    translated_text = ""
    for translation in response.translations:
        translated_text += translation.translated_text

    return translated_text


if __name__ == "__main__":
    root_dir = sys.argv[1]

    pairs = sorted(glob.glob(os.path.join(root_dir, "*")))

    for pair in pairs:

        print(pair)

        basename = os.path.basename(pair)

        src_lang, tgt_lang = basename.split("-")
        if src_lang not in flores_to_iso.keys() or tgt_lang not in flores_to_iso.keys():
            continue

        if src_lang == "eng_Latn":
            lang = tgt_lang
        else:
            lang = src_lang

        lang = flores_to_iso[lang]

        if lang not in "as bn doi gom gu hi kn mai ml mni_Mtei mr ne or pa sa sd ta te ur":
            continue

        print(f"{src_lang} - {tgt_lang}")

        # source to target translations

        src_infname = os.path.join(pair, f"test.{src_lang}")
        tgt_outfname = os.path.join(pair, f"test.{tgt_lang}.pred.google")
        if os.path.exists(src_infname) and not os.path.exists(tgt_outfname):
            src_sents = [
                sent.replace("\n", "").strip()
                for sent in open(src_infname, "r").read().split("\n")
                if sent
            ]
            translations = [
                translate_text(text, src_lang, tgt_lang).strip() for text in tqdm(src_sents)
            ]
            with open(tgt_outfname, "w") as f:
                f.write("\n".join(translations))

        # # target to source translations
        tgt_infname = os.path.join(pair, f"test.{tgt_lang}")
        src_outfname = os.path.join(pair, f"test.{src_lang}.pred.google")
        if os.path.exists(tgt_infname) and not os.path.exists(src_outfname):
            tgt_sents = [
                sent.replace("\n", "").strip()
                for sent in open(tgt_infname, "r").read().split("\n")
                if sent
            ]
            translations = [
                translate_text(text, tgt_lang, src_lang).strip() for text in tqdm(tgt_sents)
            ]

            with open(src_outfname, "w") as f:
                f.write("\n".join(translations))
