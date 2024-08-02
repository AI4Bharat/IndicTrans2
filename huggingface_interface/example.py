%%writefile IndicTrans2/huggingface_interface/example.py
import sys
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
from IndicTransTokenizer import IndicProcessor
from mosestokenizer import MosesSentenceSplitter
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA
from peft import PeftModel

en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"  # Base model checkpoint directory
lora_ckpt_dir = "/kaggle/working/output"  # Path to the fine-tuned LoRA checkpoint
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if len(sys.argv) > 1:
    quantization = sys.argv[1]
    attn_implementation = sys.argv[2]
else:
    quantization = ""
    attn_implementation = "eager"

# FLORES language code mapping to 2 letter ISO language code for compatibility
# with Indic NLP Library (https://github.com/anoopkunchukuttan/indic_nlp_library)
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
    "sat_Olck": "sat",
    "snd_Arab": "ur",
    "snd_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}

def split_sentences(input_text, lang):
    if lang == "eng_Latn":
        input_sentences = sent_tokenize(input_text)
        with MosesSentenceSplitter(flores_codes[lang]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        if len(sents_nltk) < len(sents_moses):
            input_sentences = sents_nltk
        else:
            input_sentences = sents_moses
        input_sentences = [sent.replace("\xad", "") for sent in input_sentences]
    else:
        input_sentences = sentence_split(
            input_text, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA
        )
    return input_sentences

def initialize_model_and_tokenizer(ckpt_dir, quantization, attn_implementation):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None 
        
    if attn_implementation == "flash_attention_2":
        if is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10():
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig is None:
        model = model.to(DEVICE)
        model.half()

    model.eval()

    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations

def translate_paragraph(input_text, src_lang, tgt_lang, model, tokenizer, ip):
    input_sentences = split_sentences(input_text, src_lang)
    translated_text = batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip)
    return " ".join(translated_text)

def translate_flores_dataset(input_file, output_file, src_lang, tgt_lang, model, tokenizer, ip):
    with open(input_file, 'r', encoding='utf-8') as infile:
        input_texts = infile.readlines()

    translations = []
    for input_text in input_texts:
        input_text = input_text.strip()
        if not input_text:
            continue
        translated_text = translate_paragraph(input_text, src_lang, tgt_lang, model, tokenizer, ip)
        translations.append(translated_text)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for translation in translations:
            outfile.write(translation + '\n')

ip = IndicProcessor(inference=True)

en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(
    en_indic_ckpt_dir, quantization, attn_implementation
)

lora_model = PeftModel.from_pretrained(en_indic_model, lora_ckpt_dir)

src_lang, tgt_lang = "eng_Latn", "sat_Olck"
input_file = "/kaggle/input/flores-santali-test/dev.eng_Latn"
output_file = "/kaggle/working/dev.sat_Olck"

translate_flores_dataset(input_file, output_file, src_lang, tgt_lang, lora_model, en_indic_tokenizer, ip)
print(f"Translation complete. Translated text saved to {output_file}")
