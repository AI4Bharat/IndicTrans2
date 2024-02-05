import os
import gc
import torch
import warnings
import argparse
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")

system_names = {
    "ai4bharat/indictrans2-indic-en-1B": "it2-1B-{}-hf",
    "ai4bharat/indictrans2-en-indic-1B": "it2-1B-{}-hf",
    "ai4bharat/indictrans2-indic-en-dist-200M": "it2-dist-200M-{}-hf",
    "ai4bharat/indictrans2-en-indic-dist-200M": "it2-dist-200M-{}-hf",
}


def get_direction(model_name):
    pieces = model_name.split("/")[-1].split("-")
    return f"{pieces[1]}-{pieces[2]}"


def get_arg_parser():
    parser = argparse.ArgumentParser(description="run HF MT models")
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-d", "--devtest_dir", type=str, default="")
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        default="none",
        choices=["none", "4bit", "8bit"],
    )
    parser.add_argument("--batch_size", type=int, default=512)
    return parser


def main(args):
    if args.quantization == "none":
        qconfig = None
    elif args.quantization == "8bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16
        )
    elif args.quantization == "4bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    args.batch_size = {
        "4bit": 256,
        "8bit": 256,
        "none": 128,
    }[args.quantization]

    direction = get_direction(args.model)

    tokenizer = IndicTransTokenizer(direction=direction)
    ip = IndicProcessor(inference=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, trust_remote_code=True, quantization_config=qconfig
    )

    if args.quantization == "none":
        model = model.to(device)

    system_name = system_names[args.model].format(args.quantization)

    for pair in sorted(os.listdir(args.devtest_dir)):
        l1, l2 = pair.split("-")

        if direction == "indic-en":
            src_lang = l1 if l1 != "eng_Latn" else l2
            tgt_lang = "eng_Latn"
        elif direction == "en-indic":
            tgt_lang = l1 if l1 != "eng_Latn" else l2
            src_lang = "eng_Latn"

        infname = os.path.join(args.devtest_dir, pair, f"test.{src_lang}")
        predfname = os.path.join(
            args.devtest_dir, pair, f"test.{tgt_lang}.pred.{system_name}"
        )

        if not os.path.exists(infname):
            print(f" | > Source file does not exist: {infname}")
            continue

        if os.path.exists(predfname):
            print(f" | > Prediction file exists: {predfname}")
            continue

        predictions = []

        print(f" | > Model: {args.model}")
        print(f" | > Quantization: {args.quantization}")
        print(f" | > Devtest dir: {args.devtest_dir}")
        print(f" | > Source: {src_lang}")
        print(f" | > Target: {tgt_lang}")
        print(f" | > Batch size: {args.batch_size}")

        with open(infname, "r", encoding="utf-8") as f:
            src_lines = [x.strip() for x in f.readlines()]

        for i in tqdm(range(0, len(src_lines), args.batch_size)):
            batch = src_lines[i : i + args.batch_size]
            batch = ip.preprocess_batch(batch, src_lang, tgt_lang)
            inputs = tokenizer(batch, src=True, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    num_beams=5,
                    max_length=256,
                    min_length=0,
                    num_return_sequences=1,
                )

            output_sents = tokenizer.batch_decode(outputs, src=False)
            predictions += ip.postprocess_batch(output_sents, tgt_lang)

            del batch, inputs, outputs, output_sents
            gc.collect()
            torch.cuda.empty_cache()

        with open(predfname, "w", encoding="utf-8") as f:
            f.write("\n".join(predictions))


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
