# coding=utf-8
# Copyright 2023 The IndicTrans2 Authors and AI4Bharat team. All rights reserved.
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
""" PyTorch IndicTrans config."""


from collections import OrderedDict
from typing import Any, Mapping, Optional

from transformers import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
from transformers.onnx.utils import compute_effective_axis_dimension
from transformers.utils import TensorType, is_torch_available


# Copied from transformers.models.m2m_100.configuration_m2m_100.M2M100Config->IndicTrans
class IndicTransConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IT2Model`]. It is used to instantiate an
    IT2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the IT2

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the IT2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`IT2Model`] or
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    ```"""
    model_type = "IndicTrans"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }

    def __init__(
        self,
        encoder_vocab_size=None,
        decoder_vocab_size=None,
        encoder_embed_dim=512,
        decoder_embed_dim=512,
        max_source_positions=210,
        max_target_positions=210,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        encoder_layerdrop=0.00,
        decoder_layerdrop=0.00,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="relu",
        encoder_normalize_before=False,
        decoder_normalize_before=False,
        layernorm_embedding=False,
        share_decoder_input_output_embed=False,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=True,
        decoder_start_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.encoder_normalize_before = encoder_normalize_before
        self.decoder_normalize_before = decoder_normalize_before
        self.layernorm_embedding = layernorm_embedding
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding
        self.share_decoder_input_output_embed = share_decoder_input_output_embed

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )


class IndicTransOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
            ]
        )

        if self.use_past:
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {
                0: "batch",
                1: "past_decoder_sequence + sequence",
            }
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {
                0: "batch",
                1: "decoder_sequence",
            }

        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
        return common_inputs

    # Copied from BartOnnxConfig._generate_dummy_inputs_for_sequence_classification_and_question_answering
    # A better name would be _generate_dummy_inputs_for_encoder_and_decoder because sequence classification and question
    # answering are not supported for IT2, but this name is preserved to be able to check that the copy matches what
    # was done for BART so that it can be updated if need be.
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # Copied from OnnxConfig.generate_dummy_inputs
        # Did not use super(OnnxConfigWithPast, self).generate_dummy_inputs for code clarity.
        # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
        batch_size = compute_effective_axis_dimension(
            batch_size,
            fixed_dimension=OnnxConfig.default_fixed_batch,
            num_token_to_add=0,
        )

        # If dynamic axis (-1) we forward with a fixed dimension of 8 tokens to avoid optimizations made by ONNX
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length,
            fixed_dimension=OnnxConfig.default_fixed_sequence,
            num_token_to_add=token_to_add,
        )

        # Generate dummy inputs according to compute batch and sequence
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        return common_inputs

    # Copied from transformers.models.bart.configuration_bart.BartOnnxConfig._generate_dummy_inputs_for_default_and_seq2seq_lm
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # Generate decoder inputs
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        decoder_inputs = {
            f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()
        }
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        if self.use_past:
            if not is_torch_available():
                raise ValueError(
                    "Cannot generate dummy past_keys inputs without PyTorch installed."
                )
            else:
                import torch
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            (
                num_encoder_attention_heads,
                num_decoder_attention_heads,
            ) = self.num_attention_heads
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            decoder_past_length = decoder_seq_length + 3
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                decoder_past_length,
                self._config.hidden_size // num_decoder_attention_heads,
            )

            common_inputs["decoder_attention_mask"] = torch.cat(
                [
                    common_inputs["decoder_attention_mask"],
                    torch.ones(batch, decoder_past_length),
                ],
                dim=1,
            )

            common_inputs["past_key_values"] = []
            # If the number of encoder and decoder layers are present in the model configuration, both are considered
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = (
                max(num_encoder_layers, num_decoder_layers) - min_num_layers
            )
            remaining_side_name = (
                "encoder" if num_encoder_layers > num_decoder_layers else "decoder"
            )

            for _ in range(min_num_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
            # TODO: test this.
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append(
                    (torch.zeros(shape), torch.zeros(shape))
                )
        return common_inputs

    generate_dummy_inputs = _generate_dummy_inputs_for_default_and_seq2seq_lm
