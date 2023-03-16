from typing import List, Union

import os
from os import truncate
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
from indicnlp.tokenize import sentence_tokenize

import re
import sentencepiece as spm

from inference.custom_interactive import Translator
from inference.normalize_regex_inference import normalize
from inference.flores_codes_map_indic import flores_codes
from inference.normalize_regex_inference import EMAIL_PATTERN


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
        with MosesSentenceSplitter(lang) as splitter:
            return splitter([paragraph])
    else:
        return sentence_tokenize.sentence_split(paragraph, lang=lang)


def add_token(sent: str, src_lang: str, tgt_lang: str, delimiter: str = " ") -> str:
    """
    Add special tokens indicating source and target language to the start of the input sentence.
    The resulting string will have the format: "`{src_lang} {tgt_lang} {input_sentence}`".

    Args:
        sent (str): input sentence to be translated.
        src_lang (str): language of the input sentence.
        tgt_lang (str): language in which the input sentence will be translated.
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
        src_lang (str): language of the input sentence.
        tgt_lang (str): language in which the input sentence will be translated.

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
    
    def __init__(self, ckpt_dir: str):
        """
        Initialize the model class.
        
        Args:
            ckpt_dir (str): path of the model checkpoint directory.
        """
        self.ckpt_dir = ckpt_dir
        self.en_tok = MosesTokenizer(lang="en")
        self.en_normalizer = MosesPunctNormalizer()
        self.en_detok = MosesDetokenizer(lang="en")
        self.xliterator = unicode_transliterate.UnicodeIndicTransliterator()
        
        print("Initializing sentencepiece model for SRC and TGT")
        self.sp_src = spm.SentencePieceProcessor(model_file=os.path.join(ckpt_dir, "vocab", "model.SRC"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(ckpt_dir, "vocab", "model.TGT"))
        
        print("Initializing model for translation")
        # initialize the model
        self.translator = Translator(
            data_dir=os.path.join(self.ckpt_dir, "final_bin"), 
            checkpoint_path=os.path.join(self.ckpt_dir, "model", "checkpoint_best.pt"), 
            batch_size=100
        )
    
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
            List[str]: batch of translations generated by the model.
        """
        
        assert isinstance(batch, list)
        
        # -------------------------------------------------------
        # normalize punctuations
        with open("tmp.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(batch))
        
        os.system(f"bash normalize_punctuation.sh {src_lang} < tmp.txt > tmp.txt._norm")
        
        with open("tmp.txt._norm", "r", encoding="utf-8") as f:
            batch = f.read().split("\n")
            
        os.unlink("tmp.txt")
        os.unlink("tmp.txt._norm")
        # -------------------------------------------------------
        
        preprocessed_sents = self.preprocess(batch, lang=src_lang)
        tokenized_sents = self.apply_spm(preprocessed_sents)
        tagged_sents = apply_lang_tags(tokenized_sents, src_lang, tgt_lang)
        tagged_sents = truncate_long_sentences(tagged_sents)
        
        translations = self.translator.translate(tagged_sents)
        postprocessed_sents_ = self.postprocess(translations, tgt_lang)
        
        # find the emails in the input sentences and then 
        # trim the additional spaces in the generated translations
        matches = [re.findall(EMAIL_PATTERN, x) for x in batch]
        
        postprocessed_sents = []
        for i in range(len(postprocessed_sents_)):
            sent = postprocessed_sents_[i]
            for match in matches[i]:
                potential_match = match.replace("@", "@ ")
                sent = sent.replace(potential_match, match)
            postprocessed_sents.append(sent)
        
        return postprocessed_sents
    
    
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
        
        sents = split_sentences(paragraph, src_lang)
        postprocessed_sents = self.batch_translate(sents, src_lang, tgt_lang)
        translated_paragraph = " ".join(postprocessed_sents)

        return translated_paragraph
        
        
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
            str: preprocessed input text sentence.
        """
        sent = normalize(sent)
        
        iso_lang = flores_codes[lang]
        
        transliterate = True
        if lang.split("_")[1] in ["Arab", "Olck", "Mtei", "Latn"]:
            transliterate = False
        
        pattern = r'<dnt>(.*?)</dnt>'
        raw_matches = re.findall(pattern, sent)
        
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

        processed_sent = processed_sent.replace("< dnt >", "<dnt>")
        processed_sent = processed_sent.replace("< / dnt >", "</dnt>")
        
        processed_sent_matches = re.findall(pattern, processed_sent)
        for raw_match, processed_sent_match in zip(raw_matches, processed_sent_matches):
            processed_sent = processed_sent.replace(processed_sent_match, raw_match)
    
        return processed_sent
    
    
    def preprocess(self, sents: List[str], lang: str) -> List[str]:
        """
        Preprocess a batch of input sentences for the translation.
        
        Args:
            sents (List[str]): batch of input sentences to preprocess.
            lang (str): flores language code of the input sentences.
            
        Returns:
            List[str]: preprocessed batch of input sentences.
        """
        if lang == "eng_Latn":
            processed_sents = [
                self.preprocess_sent(sent, None, lang) for sent in tqdm(sents)
            ]
        else:
            normfactory = indic_normalize.IndicNormalizerFactory()
            normalizer = normfactory.get_normalizer(flores_codes[lang])

            processed_sents = [
                self.preprocess_sent(sent, normalizer, lang) for sent in tqdm(sents)
            ]
        
        return processed_sents

    def postprocess(self, sents: List[str], lang: str, common_lang: str = "hin_Deva") -> List[str]:
        """
        Postprocesses a batch of input sentences after the translation generations.
        
        Args:
            sents (List[str]): batch of translated sentences to postprocess.
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
                    self.xliterator.transliterate(sent, flores_codes[common_lang], lang), lang
                )
                postprocessed_sents.append(outstr)
        
        return postprocessed_sents
