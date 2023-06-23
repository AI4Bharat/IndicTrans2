from typing import List, Union, Tuple

import os
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer
from sacremoses import MosesDetokenizer
import codecs
from tqdm import tqdm
from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_detokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate
from mosestokenizer import MosesSentenceSplitter
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

import re
import uuid
import hashlib
import sentencepiece as spm
from nltk.tokenize import sent_tokenize

from .normalize_regex_inference import normalize
from .flores_codes_map_indic import flores_codes, iso_to_flores
from .normalize_regex_inference import EMAIL_PATTERN
from .normalize_punctuation import punc_norm

# PWD = os.path.dirname(__file__)

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
    if lang == "eng_Latn":
        # fails to handle sentence splitting in case of
        # with MosesSentenceSplitter(lang) as splitter:
        #     return splitter([paragraph])
        return sent_tokenize(paragraph)
    else:
        return sentence_split(paragraph, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA)


def add_token(sent: str, src_lang: str, tgt_lang: str, delimiter: str = " ") -> str:
    """
    Add special tokens indicating source and target language to the start of the input sentence.
    The resulting string will have the format: "`{src_lang} {tgt_lang} {input_sentence}`".

    Args:
        sent (str): input sentence to be translated.
        src_lang (str): flores lang code of the input sentence.
        tgt_lang (str): flores lang code in which the input sentence will be translated.
        delimiter (str): separator to add between language tags and input sentence (default: " ").

    Returns:
        str: input sentence with the special tokens added to the start.
    """
    return src_lang + delimiter + tgt_lang + delimiter + sent


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
    tagged_sents = []
    for sent in sents:
        tagged_sent = add_token(sent.strip(), src_lang, tgt_lang)
        tagged_sents.append(tagged_sent)
    return tagged_sents


def truncate_long_sentences(sents: List[str]) -> List[str]:
    """
    Truncates the sentences that exceed the maximum sequence length. 
    The maximum sequence for the IndicTransv2 model is limited to 256 tokens.
    
    Args:
        sents (List[str]): list of input sentences to truncate.
    
    Returns:
        List[str]: list of truncated input sentences.
    """
    MAX_SEQ_LEN = 256
    new_sents = []

    for sent in sents:
        words = sent.split()
        num_words = len(words)
        if num_words > MAX_SEQ_LEN:
            print_str = " ".join(words[:5]) + " .... " + " ".join(words[-5:])
            sent = " ".join(words[:MAX_SEQ_LEN])
            print(
                f"WARNING: Sentence {print_str} truncated to 256 tokens as it exceeds maximum length limit"
            )

        new_sents.append(sent)
    
    return new_sents


class Model:
    """
    Model class to run the IndicTransv2 models using python interface.
    """
    
    def __init__(
        self,
        ckpt_dir: str,
        device: str = "cuda",
        input_lang_code_format: str = "flores",
        model_type: str = "ctranslate2"
    ):
        """
        Initialize the model class.
        
        Args:
            ckpt_dir (str): path of the model checkpoint directory.
            device (str, optional): where to load the model (defaults: cuda).
        """
        self.ckpt_dir = ckpt_dir
        self.en_tok = MosesTokenizer(lang="en")
        self.en_normalizer = MosesPunctNormalizer()
        self.en_detok = MosesDetokenizer(lang="en")
        self.xliterator = unicode_transliterate.UnicodeIndicTransliterator()
        
        print("Initializing sentencepiece model for SRC and TGT")
        self.sp_src = spm.SentencePieceProcessor(model_file=os.path.join(ckpt_dir, "vocab", "model.SRC"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(ckpt_dir, "vocab", "model.TGT"))

        self.input_lang_code_format = input_lang_code_format
        
        print("Initializing model for translation")
        # initialize the model
        if model_type == "ctranslate2":
            import ctranslate2
            self.translator = ctranslate2.Translator(self.ckpt_dir, device=device)#, compute_type="auto")
            self.translate_lines = self.ctranslate2_translate_lines
        elif model_type == "fairseq":
            from .custom_interactive import Translator
            self.translator = Translator(
                data_dir=os.path.join(self.ckpt_dir, "final_bin"), 
                checkpoint_path=os.path.join(self.ckpt_dir, "model", "checkpoint_best.pt"), 
                batch_size=100
            )
            self.translate_lines = self.fairseq_translate_lines
        else:
            raise NotImplementedError(f"Unknown model_type: {model_type}")
    
    def ctranslate2_translate_lines(self, lines: List[str]) -> List[str]:
        tokenized_sents = [x.strip().split(" ") for x in lines]
        translations = self.translator.translate_batch(
            tokenized_sents,
            max_batch_size=9216,
            batch_type="tokens",
            max_input_length=160,
            max_decoding_length=256,
            beam_size=5,
        )
        translations = [" ".join(x.hypotheses[0]) for x in translations]
        return translations
    
    def fairseq_translate_lines(self, lines: List[str]) -> List[str]:
        return self.translator.translate(lines)
    
    def paragraphs_batch_translate__multilingual(self, batch_payloads: List[tuple]) -> List[str]:
        """
        Translates a batch of input paragraphs (including pre/post processing) 
        from any language to any language.
        
        Args:
            batch_payloads (List[tuple]): batch of long input-texts to be translated, each in format: (paragraph, src_lang, tgt_lang)
        
        Returns:
            List[str]: batch of paragraph-translations in the respective languages.
        """
        paragraph_id_to_sentence_range = []
        global__sents = []
        global__preprocessed_sents = []
        global__preprocessed_sents_placeholder_entity_map = []
        
        for i in range(len(batch_payloads)):
            paragraph, src_lang, tgt_lang = batch_payloads[i]
            if self.input_lang_code_format == "iso":
                src_lang, tgt_lang = iso_to_flores[src_lang], iso_to_flores[tgt_lang]
            
            batch = split_sentences(paragraph, src_lang)
            global__sents.extend(batch)

            preprocessed_sents, placeholder_entity_map_sents = self.preprocess_batch(batch, src_lang, tgt_lang)

            global_sentence_start_index = len(global__preprocessed_sents)
            global__preprocessed_sents.extend(preprocessed_sents)
            global__preprocessed_sents_placeholder_entity_map.extend(placeholder_entity_map_sents)
            paragraph_id_to_sentence_range.append((global_sentence_start_index, len(global__preprocessed_sents)))
        
        translations = self.translate_lines(global__preprocessed_sents)

        translated_paragraphs = []
        for paragraph_id, sentence_range in enumerate(paragraph_id_to_sentence_range):
            tgt_lang = batch_payloads[paragraph_id][2]
            if self.input_lang_code_format == "iso":
                tgt_lang = iso_to_flores[tgt_lang]
            
            postprocessed_sents = self.postprocess_batch(
                translations[sentence_range[0]:sentence_range[1]],
                tgt_lang,
                input_sents = global__sents[sentence_range[0]:sentence_range[1]],
                placeholder_entity_map = global__preprocessed_sents_placeholder_entity_map[sentence_range[0]:sentence_range[1]]
            )
            translated_paragraph = " ".join(postprocessed_sents)
            translated_paragraphs.append(translated_paragraph)
        
        return translated_paragraphs

    # translate a batch of sentences from src_lang to tgt_lang
    def batch_translate(self, batch: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Translates a batch of input sentences (including pre/post processing) 
        from source language to target language.
        
        Args:
            batch (List[str]): batch of input sentences to be translated.
            src_lang (str): flores source language code.
            tgt_lang (str): flores target language code.
        
        Returns:
            List[str]: batch of translated-sentences generated by the model.
        """
        
        assert isinstance(batch, list)

        if self.input_lang_code_format == "iso":
            src_lang, tgt_lang = iso_to_flores[src_lang], iso_to_flores[tgt_lang]

        preprocessed_sents, placeholder_entity_map_sents = self.preprocess_batch(batch, src_lang, tgt_lang)
        translations = self.translate_lines(preprocessed_sents)
        return self.postprocess_batch(translations, tgt_lang, input_sents=batch, placeholder_entity_map=placeholder_entity_map_sents)
    
    # translate a paragraph from src_lang to tgt_lang
    def translate_paragraph(self, paragraph: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translates an input text paragraph (including pre/post processing) 
        from source language to target language.
        
        Args:
            paragraph (str): input text paragraph to be translated.
            src_lang (str): flores source language code.
            tgt_lang (str): flores target language code.
        
        Returns:
            str: paragraph translation generated by the model.
        """
        
        assert isinstance(paragraph, str)
        
        if self.input_lang_code_format == "iso":
            flores_src_lang = iso_to_flores[src_lang]
        else:
            flores_src_lang = src_lang

        sents = split_sentences(paragraph, flores_src_lang)
        postprocessed_sents = self.batch_translate(sents, src_lang, tgt_lang)
        translated_paragraph = " ".join(postprocessed_sents)

        return translated_paragraph
    
    def preprocess_batch(self, batch: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Preprocess an array of sentences by normalizing, tokenization, and possibly transliterating it. It also tokenizes the 
        normalized text sequences using sentence piece tokenizer and also adds language tags.

        Args:
            batch (List[str]): input list of sentences to preprocess.
            src_lang (str): flores language code of the input text sentences.
            tgt_lang (str): flores language code of the output text sentences.
            
        Returns:
            Tuple[List[str], List[dict]]: a tuple of list of preprocessed input text sentences and also a corresponding list of dictionary 
                mapping placeholders to their original values.
        """
        preprocessed_sents, placeholder_entity_map_sents = self.preprocess(batch, lang=src_lang)
        tokenized_sents = self.apply_spm(preprocessed_sents)
        tagged_sents = apply_lang_tags(tokenized_sents, src_lang, tgt_lang)
        tagged_sents = truncate_long_sentences(tagged_sents)
        
        return tagged_sents, placeholder_entity_map_sents
        
    def apply_spm(self, sents: List[str]) -> List[str]:
        """
        Applies sentence piece encoding to the batch of input sentences.
        
        Args:
            sents (List[str]): batch of the input sentences.
        
        Returns:
            List[str]: batch of encoded sentences with sentence piece model
        """
        return [" ".join(self.sp_src.encode(sent, out_type=str)) for sent in sents]
        
    
    def preprocess_sent(
        self, 
        sent: str, 
        normalizer: Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory], 
        lang: str
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
        if lang.split("_")[1] in ["Arab", "Olck", "Mtei", "Latn"]:
            transliterate = False
        
        if iso_lang == "en":
            processed_sent = " ".join(
                self.en_tok.tokenize(
                    self.en_normalizer.normalize(sent.strip()), escape=False
                )
            )
        elif transliterate:
            # transliterates from the any specific language to devanagari
            # which is why we specify lang2_code as "hi".
            processed_sent = unicode_transliterate.UnicodeIndicTransliterator.transliterate(
                " ".join(indic_tokenize.trivial_tokenize(normalizer.normalize(sent.strip()), iso_lang)),
                iso_lang,
                "hi",
            ).replace(" ् ", "्")
        else:
            # we only need to transliterate for joint training
            processed_sent = " ".join(
                indic_tokenize.trivial_tokenize(normalizer.normalize(sent.strip()), iso_lang)
            )
    
        return processed_sent, placeholder_entity_map
    
    
    def preprocess(self, sents: List[str], lang: str):
        """
        Preprocess an array of sentences by normalizing, tokenization, and possibly transliterating it.

        Args:
            batch (List[str]): input list of sentences to preprocess.
            lang (str): flores language code of the input text sentences.
            
        Returns:
            Tuple[List[str], List[dict]]: a tuple of list of preprocessed input text sentences and also a corresponding list of dictionary 
                mapping placeholders to their original values.
        """
        processed_sents, placeholder_entity_map_sents = [], []

        if lang == "eng_Latn":
            normalizer = None
        else:
            normfactory = indic_normalize.IndicNormalizerFactory()
            normalizer = normfactory.get_normalizer(flores_codes[lang])
        
        for sent in sents:
            sent, placeholder_entity_map = self.preprocess_sent(sent, normalizer, lang)
            processed_sents.append(sent)
            placeholder_entity_map_sents.append(placeholder_entity_map)
                        
        return processed_sents, placeholder_entity_map_sents
    
    def postprocess_batch(self, translations: List[str], lang: str, input_sents: List[str] = None, placeholder_entity_map: List[dict] = None) -> List[str]:
        """
        Wrapper function over `postprocess` that postprocesses a batch of translations.
        
        Args:
            translations (List[str]): batch of translated sentences to postprocess.
            lang (str): flores language code of the input sentences.
            input_sents (List[str]): list of input text sentences that are translated (defaults: None).
            placeholder_entity_map (List[dict]): dictionary mapping placeholders to the original entity values (defaults: None).
        
        Returns:
            List[str]: postprocessed batch of input sentences.
        """
        postprocessed_sents = self.postprocess(translations, placeholder_entity_map, lang)
        return postprocessed_sents

    def postprocess(self, sents: List[str], placeholder_entity_map: List[dict], lang: str, common_lang: str = "hin_Deva") -> List[str]:
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
        sents = [self.sp_tgt.decode(x.split(" ")) for x in sents]
        
        postprocessed_sents = []
        
        if lang == "eng_Latn":
            for sent in sents:
                # outfile.write(en_detok.detokenize(sent.split(" ")) + "\n")
                postprocessed_sents.append(self.en_detok.detokenize(sent.split(" ")))
        else:
            for sent in sents:
                outstr = indic_detokenize.trivial_detokenize(
                    self.xliterator.transliterate(sent, flores_codes[common_lang], flores_codes[lang]), flores_codes[lang]
                )
                postprocessed_sents.append(outstr)
        
        assert len(postprocessed_sents) == len(placeholder_entity_map)
            
        for i in range(0, len(postprocessed_sents)):
            for key in placeholder_entity_map[i].keys():
                postprocessed_sents[i] = postprocessed_sents[i].replace(key, placeholder_entity_map[i][key])
        return postprocessed_sents
