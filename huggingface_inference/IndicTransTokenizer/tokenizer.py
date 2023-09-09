import os
import json
import torch
import sentencepiece
from typing import Dict, List, Tuple, Union

_PATH = os.path.dirname(os.path.realpath(__file__))


class IndicTransTokenizer:
    def __init__(
        self,
        src_vocab_fp=None,
        tgt_vocab_fp=None,
        src_spm_fp=None,
        tgt_spm_fp=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        direction="indic-en",
        device="cpu",
        model_max_length=256,
    ):
        self.model_max_length = model_max_length

        self.supported_langs = [
            "asm_Beng",
            "ben_Beng",
            "brx_Deva",
            "doi_Deva",
            "eng_Latn",
            "gom_Deva",
            "guj_Gujr",
            "hin_Deva",
            "kan_Knda",
            "kas_Arab",
            "kas_Deva",
            "mai_Deva",
            "mal_Mlym",
            "mar_Deva",
            "mni_Beng",
            "mni_Mtei",
            "npi_Deva",
            "ory_Orya",
            "pan_Guru",
            "san_Deva",
            "sat_Olck",
            "snd_Arab",
            "snd_Deva",
            "tam_Taml",
            "tel_Telu",
            "urd_Arab",
        ]

        self.src_vocab_fp = (
            src_vocab_fp
            if (src_vocab_fp is not None)
            else os.path.join(_PATH, direction, "dict.SRC.json")
        )
        self.tgt_vocab_fp = (
            tgt_vocab_fp
            if (tgt_vocab_fp is not None)
            else os.path.join(_PATH, direction, "dict.TGT.json")
        )
        self.src_spm_fp = (
            src_spm_fp
            if (src_spm_fp is not None)
            else os.path.join(_PATH, direction, "model.SRC")
        )
        self.tgt_spm_fp = (
            tgt_spm_fp
            if (tgt_spm_fp is not None)
            else os.path.join(_PATH, direction, "model.TGT")
        )

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token

        self.device = device

        self.encoder = self._load_json(self.src_vocab_fp)
        if self.unk_token not in self.encoder:
            raise KeyError("<unk> token must be in vocab")
        assert self.pad_token in self.encoder
        self.encoder_rev = {v: k for k, v in self.encoder.items()}

        self.decoder = self._load_json(self.tgt_vocab_fp)
        if self.unk_token not in self.encoder:
            raise KeyError("<unk> token must be in vocab")
        assert self.pad_token in self.encoder
        self.decoder_rev = {v: k for k, v in self.decoder.items()}

        # load SentencePiece model for pre-processing
        self.src_spm = self._load_spm(self.src_spm_fp)
        self.tgt_spm = self._load_spm(self.tgt_spm_fp)

    def get_vocab_size(self, src: bool) -> int:
        """Returns the size of the vocabulary"""
        return len(self.encoder) if src else len(self.decoder)

    def _load_spm(self, path: str) -> sentencepiece.SentencePieceProcessor:
        return sentencepiece.SentencePieceProcessor(model_file=path)

    def _save_json(self, data, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_json(self, path: str) -> Union[Dict, List]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _convert_token_to_id(self, token: str, src: bool) -> int:
        """Converts an token (str) into an index (integer) using the source/target vocabulary map."""
        return (
            self.encoder.get(token, self.encoder[self.unk_token])
            if src
            else self.decoder.get(token, self.encoder[self.unk_token])
        )

    def _convert_id_to_token(self, index: int, src: bool) -> str:
        """Converts an index (integer) into a token (str) using the source/target vocabulary map."""
        return (
            self.encoder_rev.get(index, self.unk_token)
            if src
            else self.decoder_rev.get(index, self.unk_token)
        )

    def _convert_tokens_to_string(self, tokens: List[str], src: bool) -> str:
        """Uses sentencepiece model for detokenization"""
        if src:
            if tokens[0] in self.supported_langs and tokens[1] in self.supported_langs:
                tokens = tokens[2:]
            return " ".join(tokens)
        else:
            return " ".join(tokens)

    def _remove_translation_tags(self, text: str) -> Tuple[List, str]:
        """Removes the translation tags before text normalization and tokenization."""
        tokens = text.split(" ")
        tags = tokens[:2]
        return tags, " ".join(tokens[2:])

    def _tokenize_src_line(self, line: str) -> List[str]:
        """Tokenizes a source line."""
        tags, text = self._remove_translation_tags(line)
        tokens = self.src_spm.encode(text, out_type=str)
        return tags + tokens

    def _tokenize_tgt_line(self, line: str) -> List[str]:
        """Tokenizes a target line."""
        return self.tgt_spm.encode(line, out_type=str)

    def tokenize(self, text: str, src: bool) -> List[str]:
        """Tokenizes a string into tokens using the source/target vocabulary."""
        return self._tokenize_src_line(text) if src else self._tokenize_tgt_line(text)

    def batch_tokenize(self, batch: List[str], src: bool) -> List[List[str]]:
        """Tokenizes a list of strings into tokens using the source/target vocabulary."""
        if not isinstance(batch, list):
            assert f"batch should be a list of strings but was given {type(batch)}"
        return [self.tokenize(line, src) for line in batch]

    def __call__(
        self,
        batch: Union[list, str],
        src: bool,
        truncation: bool = False,
        padding: str = "longest",
        max_length: int = None,
        return_tensors=None,
    ) -> List[str]:
        """Tokenizes a string into tokens and also converts them into integers using source/target vocabulary map."""
        assert padding in [
            "longest",
            "max_length",
        ], "padding should be either 'longest' or 'max_length'"

        if isinstance(batch, str):
            batch = [batch]

        batch = self.batch_tokenize(batch, src)

        if truncation and max_length is not None:
            batch = [ids[:max_length] for ids in batch]

        if padding == "longest":
            max_seq_len = max([len(ids) for ids in batch])
        else:
            max_seq_len = max_length

        attention_mask = [
            (([0] * (max_seq_len - len(ids))) + ([1] * (len(ids) + 1))) for ids in batch
        ]

        batch = [
            (
                ([self.pad_token] * (max_seq_len - len(tokens)))
                + tokens
                + [self.eos_token]
            )
            for tokens in batch
        ]

        input_ids = [
            [self._convert_token_to_id(token, src) for token in tokens]
            for tokens in batch
        ]

        encodings = {"input_ids": input_ids, "attention_mask": attention_mask}

        if return_tensors:
            # returns tensors placed on the appropriate device
            encodings = {
                k: torch.tensor(v, dtype=torch.int64, device=self.device)
                for (k, v) in encodings.items()
            }

        return encodings

    def batch_decode(
        self, batch: Union[list, torch.Tensor], src: bool
    ) -> List[List[str]]:
        """Detokenizes a list of integer ids or a tensor into a list of strings using the source/target vocabulary."""
        is_special_token = lambda x: (
            (x == self.pad_token) or (x == self.bos_token) or (x == self.eos_token)
        )

        if isinstance(batch, torch.Tensor):
            batch = batch.tolist()

        batch = [[self._convert_id_to_token(_id, src) for _id in ids] for ids in batch]
        batch = [
            [token for token in tokens if not is_special_token(token)]
            for tokens in batch
        ]
        batch = [self._convert_tokens_to_string(tokens, src) for tokens in batch]

        return batch
