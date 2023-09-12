# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
import torch.nn as nn

from configuration_indictrans import IndicTransConfig
from modeling_indictrans import IndicTransForConditionalGeneration


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.data
    return lin_layer


def convert_fairseq_IT2_checkpoint_from_disk(checkpoint_path):
    model = torch.load(checkpoint_path, map_location="cpu")
    args = model["args"] or model["cfg"]["model"]
    state_dict = model["model"]
    remove_ignore_keys_(state_dict)
    encoder_vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]
    decoder_vocab_size = state_dict["decoder.embed_tokens.weight"].shape[0]

    config = IndicTransConfig(
        encoder_vocab_size=encoder_vocab_size,
        decoder_vocab_size=decoder_vocab_size,
        max_source_positions=args.max_source_positions,
        max_target_positions=args.max_target_positions,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        layernorm_embedding=args.layernorm_embedding,
        encoder_normalize_before=args.encoder_normalize_before,
        decoder_normalize_before=args.decoder_normalize_before,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_embed_dim,
        decoder_ffn_dim=args.decoder_ffn_embed_dim,
        encoder_embed_dim=args.encoder_embed_dim,
        decoder_embed_dim=args.decoder_embed_dim,
        encoder_layerdrop=args.encoder_layerdrop,
        decoder_layerdrop=args.decoder_layerdrop,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function=args.activation_fn,
        share_decoder_input_output_embed=args.share_decoder_input_output_embed,
        scale_embedding=not args.no_scale_embedding,
    )

    model = IndicTransForConditionalGeneration(config)
    model.model.load_state_dict(state_dict, strict=False)
    if not args.share_decoder_input_output_embed:
        model.lm_head = make_linear_from_emb(
            state_dict["decoder.output_projection.weight"]
        )
    print(model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq_path",
        default="indic-en/model/checkpoint_best.pt",
        type=str,
        help="path to a model.pt on local filesystem.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="indic-en/hf_model",
        type=str,
        help="Path to the output PyTorch model.",
    )

    args = parser.parse_args()
    model = convert_fairseq_IT2_checkpoint_from_disk(args.fairseq_path)
    model.save_pretrained(args.pytorch_dump_folder_path)
