import os
import sys
import glob
import requests
from urllib.parse import urlencode
from dotenv import dotenv_values
import traceback
import time

flores_to_iso = {
    "asm_Beng": "as",
    "ben_Beng": "bn",
    "brx_Deva": "brx",
    "doi_Deva": "doi",
    "eng_Latn": "en",
    "gom_Deva": "gom",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ks",
    "kas_Deva": "ks_Deva",
    "mai_Deva": "mai",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "mni_Beng",
    "mni_Mtei": "mni",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "sa",
    "sat_Olck": "sat",
    "snd_Arab": "sd",
    "snd_Deva": "sd_Deva",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}


class AzureTranslator:
    def __init__(
        self,
        subscription_key: str,
        region: str,
        endpoint: str = "https://api.cognitive.microsofttranslator.com",
    ) -> None:
        self.http_headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Ocp-Apim-Subscription-Region": region,
        }
        self.translate_endpoint = endpoint + "/translate?api-version=3.0&"
        self.languages_endpoint = endpoint + "/languages?api-version=3.0"

        self.supported_languages = self.get_supported_languages()

    def get_supported_languages(self) -> dict:
        return requests.get(self.languages_endpoint).json()["translation"]

    def batch_translate(self, texts: list, src_lang: str, tgt_lang: str) -> list:
        if not texts:
            return texts

        src_lang = flores_to_iso[src_lang]
        tgt_lang = flores_to_iso[tgt_lang]

        if src_lang not in self.supported_languages:
            raise NotImplementedError(
                f"Source language code: `{src_lang}` not supported!"
            )

        if tgt_lang not in self.supported_languages:
            raise NotImplementedError(
                f"Target language code: `{tgt_lang}` not supported!"
            )

        body = [{"text": text} for text in texts]
        query_string = urlencode(
            {
                "from": src_lang,
                "to": tgt_lang,
            }
        )

        try:
            response = requests.post(
                self.translate_endpoint + query_string,
                headers=self.http_headers,
                json=body,
            )
        except:
            traceback.print_exc()
            return None

        try:
            response = response.json()
        except:
            traceback.print_exc()
            print("Response:", response.text)
            return None

        return [payload["translations"][0]["text"] for payload in response]

    def text_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return self.batch_translate([text], src_lang, tgt_lang)[0]


if __name__ == "__main__":
    root_dir = sys.argv[1]
    
    # Expects a .env file containing the API credentials.
    config = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))

    t = AzureTranslator(
        config["AZURE_TRANSLATOR_TEXT_SUBSCRIPTION_KEY"],
        config["AZURE_TRANSLATOR_TEXT_REGION"],
        config["AZURE_TRANSLATOR_TEXT_ENDPOINT"],
    )

    pairs = sorted(glob.glob(os.path.join(root_dir, "*")))

    for i, pair in enumerate(pairs):
        basename = os.path.basename(pair)
        
        print(pair)
    
        src_lang, tgt_lang = basename.split("-")

        print(f"{src_lang} - {tgt_lang}")

        # source to target translations
        src_infname = os.path.join(pair, f"test.{src_lang}")
        tgt_outfname = os.path.join(pair, f"test.{tgt_lang}.pred.azure")
        if not os.path.exists(src_infname):
            continue

        src_sents = [
            sent.replace("\n", "").strip()
            for sent in open(src_infname, "r").read().split("\n")
            if sent
        ]

        if not os.path.exists(tgt_outfname):
            try:
                translations = []
                for i in range(0, len(src_sents), 128):
                    start, end = i, int(min(i + 128, len(src_sents)))
                    translations.extend(
                        t.batch_translate(src_sents[start:end], src_lang, tgt_lang)
                    )
                with open(tgt_outfname, "w") as f:
                    f.write("\n".join(translations))
                
                time.sleep(10)
            except Exception as e:
                print(e)
                continue

        # target to source translations
        tgt_infname = os.path.join(pair, f"test.{tgt_lang}")
        src_outfname = os.path.join(pair, f"test.{src_lang}.pred.azure")
        if not os.path.exists(tgt_infname):
            continue

        tgt_sents = [
            sent.replace("\n", "").strip()
            for sent in open(tgt_infname, "r").read().split("\n")
            if sent
        ]

        if not os.path.exists(src_outfname):
            try:
                translations = []
                for i in range(0, len(tgt_sents), 128):
                    start, end = i, int(min(i + 128, len(tgt_sents)))
                    translations.extend(
                        t.batch_translate(tgt_sents[start:end], tgt_lang, src_lang)
                    )
                with open(src_outfname, "w") as f:
                    f.write("\n".join(translations))
            except Exception as e:
                continue

            time.sleep(10)
