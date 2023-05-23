import os
import re
import sys
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# dictionary mapping flores codes to M2M-100 supported codes
langs_supported = {
    "eng_Latn": "en",
    "ben_Beng": "bn",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "snd_Arab": "sd",
    "tam_Taml": "ta",
    "urd_Arab": "ur",
}


def predict(batch, tokenizer, model, bos_token_id):
    encoded_batch = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
    generated_tokens = model.generate(
        **encoded_batch,
        num_beams=5,
        max_length=256,
        min_length=0,
        forced_bos_token_id=bos_token_id,
    )
    hypothesis = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return hypothesis


def main(devtest_data_dir, batch_size):
    # load the pre-trained M2M-100 tokenizer and model
    model_name = "facebook/m2m100-12B-last-ckpt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    # iterate over a list of language pairs from `devtest_data_dir`
    for pair in sorted(os.listdir(devtest_data_dir)):
        if "-" not in pair:
            continue

        src_lang, tgt_lang = pair.split("-")

        # check if the source and target languages are supported
        if (
            src_lang not in langs_supported.keys()
            or tgt_lang not in langs_supported.keys()
        ):
            print(f"Skipping {src_lang}-{tgt_lang} ...")
            continue

        # -------------------------------------------------------------------
        #                   source to target evaluation
        # -------------------------------------------------------------------
        print(f"Evaluating {src_lang}-{tgt_lang} ...")

        infname = os.path.join(devtest_data_dir, pair, f"test.{src_lang}")
        outfname = os.path.join(devtest_data_dir, pair, f"test.{tgt_lang}.pred.m2m100")

        with open(infname, "r") as f:
            src_sents = f.read().split("\n")

        add_new_line = False
        if src_sents[-1] == "":
            add_new_line = True
            src_sents = src_sents[:-1]

        # set the source language for tokenization
        tokenizer.src_lang = langs_supported[src_lang]

        # process sentences in batches and generate predictions
        hypothesis = []
        for i in tqdm(range(0, len(src_sents), batch_size)):
            start, end = i, int(min(len(src_sents), i + batch_size))
            batch = src_sents[start:end]
            bos_token_id = tokenizer.lang_code_to_id[langs_supported[tgt_lang]]
            hypothesis += predict(batch, tokenizer, model, bos_token_id)

        assert len(hypothesis) == len(src_sents)

        hypothesis = [
            re.sub("\s+", " ", x.replace("\n", " ").replace("\t", " ")).strip()
            for x in hypothesis
        ]
        if add_new_line:
            hypothesis = hypothesis

        with open(outfname, "w") as f:
            f.write("\n".join(hypothesis))

        # -------------------------------------------------------------------
        #                   target to source evaluation
        # -------------------------------------------------------------------
        infname = os.path.join(devtest_data_dir, pair, f"test.{tgt_lang}")
        outfname = os.path.join(devtest_data_dir, pair, f"test.{src_lang}.pred.m2m100")

        with open(infname, "r") as f:
            src_sents = f.read().split("\n")

        add_new_line = False
        if src_sents[-1] == "":
            add_new_line = True
            src_sents = src_sents[:-1]

        # set the source language for tokenization
        tokenizer.src_lang = langs_supported[tgt_lang]

        # process sentences in batches and generate predictions
        hypothesis = []
        for i in tqdm(range(0, len(src_sents), batch_size)):
            start, end = i, int(min(len(src_sents), i + batch_size))
            batch = src_sents[start:end]
            bos_token_id = tokenizer.lang_code_to_id[langs_supported[src_lang]]
            hypothesis += predict(batch, tokenizer, model, bos_token_id)

        assert len(hypothesis) == len(src_sents)

        hypothesis = [
            re.sub("\s+", " ", x.replace("\n", " ").replace("\t", " ")).strip()
            for x in hypothesis
        ]
        if add_new_line:
            hypothesis = hypothesis

        with open(outfname, "w") as f:
            f.write("\n".join(hypothesis))


if __name__ == "__main__":
    # expects En-X subdirectories pairs within the devtest data directory
    devtest_data_dir = sys.argv[1]
    batch_size = int(sys.argv[2])

    if not torch.cuda.is_available():
        print("No GPU available")
        sys.exit(1)

    main(devtest_data_dir, batch_size)
