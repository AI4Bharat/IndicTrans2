import sys
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"  # ai4bharat/indictrans2-en-indic-dist-200M
indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"  # ai4bharat/indictrans2-indic-en-dist-200M
indic_indic_ckpt_dir = "ai4bharat/indictrans2-indic-indic-1B"  # ai4bharat/indictrans2-indic-indic-dist-320M
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if len(sys.argv) > 1:
    quantization = sys.argv[1]
else:
    quantization = ""


def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
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

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
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
            src=True,
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
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


ip = IndicProcessor(inference=True)

en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)
indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir, "indic-en", quantization)
indic_indic_tokenizer, indic_indic_model = initialize_model_and_tokenizer(indic_indic_ckpt_dir, "indic-indic", quantization)

# ---------------------------------------------------------------------------
#                              Hindi to English
# ---------------------------------------------------------------------------
hi_sents = [
    "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
    "उसके पास बहुत सारी पुरानी किताबें हैं, जिन्हें उसने अपने दादा-परदादा से विरासत में पाया।",
    "मुझे समझ में नहीं आ रहा कि मैं अपनी समस्या का समाधान कैसे ढूंढूं।",
    "वह बहुत मेहनती और समझदार है, इसलिए उसे सभी अच्छे मार्क्स मिले।",
    "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
    "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
    "वह अपनी दीदी के साथ बाजार गयी थी ताकि वह नई साड़ी खरीद सके।",
    "राज ने मुझसे कहा कि वह अगले महीने अपनी नानी के घर जा रहा है।",
    "सभी बच्चे पार्टी में मज़ा कर रहे थे और खूब सारी मिठाइयाँ खा रहे थे।",
    "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
]
src_lang, tgt_lang = "hin_Deva", "eng_Latn"
en_translations = batch_translate(hi_sents, src_lang, tgt_lang, indic_en_model, indic_en_tokenizer, ip)

print(f"\n{src_lang} - {tgt_lang}")
for input_sentence, translation in zip(hi_sents, en_translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")


# ---------------------------------------------------------------------------
#                              English to Hindi
# ---------------------------------------------------------------------------
en_sents = [
    "When I was young, I used to go to the park every day.",
    "He has many old books, which he inherited from his ancestors.",
    "I can't figure out how to solve my problem.",
    "She is very hardworking and intelligent, which is why she got all the good marks.",
    "We watched a new movie last week, which was very inspiring.",
    "If you had met me at that time, we would have gone out to eat.",
    "She went to the market with her sister to buy a new sari.",
    "Raj told me that he is going to his grandmother's house next month.",
    "All the kids were having fun at the party and were eating lots of sweets.",
    "My friend has invited me to his birthday party, and I will give him a gift.",
]
src_lang, tgt_lang = "eng_Latn", "hin_Deva"
hi_translations = batch_translate(en_sents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

print(f"\n{src_lang} - {tgt_lang}")
for input_sentence, translation in zip(en_sents, hi_translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")


# ---------------------------------------------------------------------------
#                              Hindi to Marathi
# ---------------------------------------------------------------------------
hi_sents = [
    "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
    "उसके पास बहुत सारी पुरानी किताबें हैं, जिन्हें उसने अपने दादा-परदादा से विरासत में पाया।",
    "मुझे समझ में नहीं आ रहा कि मैं अपनी समस्या का समाधान कैसे ढूंढूं।",
    "वह बहुत मेहनती और समझदार है, इसलिए उसे सभी अच्छे मार्क्स मिले।",
    "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
    "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
    "वह अपनी दीदी के साथ बाजार गयी थी ताकि वह नई साड़ी खरीद सके।",
    "राज ने मुझसे कहा कि वह अगले महीने अपनी नानी के घर जा रहा है।",
    "सभी बच्चे पार्टी में मज़ा कर रहे थे और खूब सारी मिठाइयाँ खा रहे थे।",
    "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
]
src_lang, tgt_lang = "hin_Deva", "mar_Deva"
mr_translations = batch_translate(hi_sents, src_lang, tgt_lang, indic_indic_model, indic_indic_tokenizer, ip)

print(f"\n{src_lang} - {tgt_lang}")
for input_sentence, translation in zip(hi_sents, mr_translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")
