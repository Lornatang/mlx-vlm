# Copyright Larry. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    """Hyperparameters for the language-model (LLM) backbone.

    Attributes:
        model_type (str): Identifier for the language-model architecture.
        hidden_size (int, optional): Dimensionality of the LLM embeddings. Defaults to 896.
        num_hidden_layers (int): Number of transformer layers in the LLM.
        intermediate_size (int): Size of the feed-forward inner projection.
        num_attention_heads (int): Number of query heads in self-attention.
        rms_norm_eps (float): Epsilon used in RMSNorm for numerical stability.
        vocab_size (int): Size of the token vocabulary.
        num_key_value_heads (int): Number of key/value heads for grouped-query attention.
        max_position_embeddings (int): Maximum sequence length the position encodings support.
        rope_theta (float): Base frequency for rotary position embeddings (RoPE).
        rope_traditional (bool): Whether to use the traditional (non-NeoX) RoPE interleaving.
        rope_scaling (Optional[Dict[str, Union[float, str]]], optional): Optional RoPE scaling strategy and parameters. Defaults to ``None``.
        head_dim (int): Dimensionality of each attention head.
        tie_word_embeddings (bool): Whether to tie the input and output embedding weights.
    """

    model_type: str
    hidden_size: int = 896
    num_hidden_layers: int = 24
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    rms_norm_eps: float = 1e-06
    vocab_size: int = 151936
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    head_dim: int = 128
    tie_word_embeddings: bool = True


@dataclass
class VisionConfig(BaseModelConfig):
    """Hyperparameters for the FastViTHD vision tower.

    Attributes:
        model_type (str, optional): Identifier for the vision encoder architecture. Defaults to ``"jingyu"``.
        hidden_size (int, optional): Dimensionality of the projected visual features. Defaults to 1024.
        intermediate_size (int, optional): Size of the feed-forward inner projection in vision blocks. Defaults to 3072.
        image_size (int, optional): Resolution (height/width) of the input images. Defaults to 1024.
        patch_size (int, optional): Width/height of each patch embedding. Defaults to 64.
        projection_dim (int, optional): Dimensionality of the projector output shared with the LLM. Defaults to 768.
        num_classes (int, optional): Number of classes of the classification head (unused). Defaults to 1000.
        down_patch_size (int, optional): Patch size of the downsampling aggregation layer. Defaults to 7.
        down_stride (int, optional): Stride of the downsampling aggregation layer. Defaults to 2.
        layer_scale_init_value (float, optional): Initial scale applied to residual branches. Defaults to 1e-5.
        cls_ratio (float, optional): Ratio determining the size of the learned class token. Defaults to 2.0.
        layers (list[int], optional): Number of blocks in each FastViTHD stage. Defaults to ``[2, 12, 24, 4, 2]``.
        embed_dims (list[int], optional): Channel dimensionality of each FastViTHD stage. Defaults to ``[96, 192, 384, 768, 1536]``.
        mlp_ratios (list[int], optional): MLP expansion ratio per stage. Defaults to ``[4, 4, 4, 4, 4]``.
        downsamples (list[bool], optional): Whether each stage applies spatial downsampling.
            Defaults to ``[True, True, True, True, True]``.
        pos_embs_shapes (list[Optional[tuple[int, int]]], optional): Positional embedding shape per stage
            (``None`` if unused). Defaults to ``[None, None, None, (7, 7), (7, 7)]``.
        token_mixers (tuple[str, ...], optional): Token-mixing type per stage (repmixer or attention).
            Defaults to ``("repmixer", "repmixer", "repmixer", "attention", "attention")``.
        repmixer_kernel_size (int, optional): Kernel size of the reparametrizable mixer. Defaults to 3.
    """

    model_type: str = "jingyu"
    hidden_size: int = 1024
    intermediate_size: int = 3072
    image_size: int = 1024
    patch_size: int = 64
    projection_dim: int = 768
    num_classes = 1000
    down_patch_size = 7
    down_stride = 2
    layer_scale_init_value = 1e-5
    cls_ratio = 2.0
    # FastViTHD variant
    layers = [2, 12, 24, 4, 2]
    embed_dims = [96, 192, 384, 768, 1536]
    mlp_ratios = [4, 4, 4, 4, 4]
    downsamples = [True, True, True, True, True]
    pos_embs_shapes = [None, None, None, (7, 7), (7, 7)]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention", "attention")
    repmixer_kernel_size = 3


@dataclass
class ModelConfig(BaseModelConfig):
    """Top-level configuration combining the LLM and vision tower.

    Attributes:
        text_config (TextConfig): Configuration of the language-model backbone.
        vision_config (VisionConfig): Configuration of the vision tower.
        model_type (str, optional): Identifier for the merged architecture. Defaults to ``"jingyu"``.
        ignore_index (int, optional): Target label excluded from the loss computation. Defaults to -100.
        image_token_index (int, optional): Token ID reserved as the image placeholder. Defaults to -200.
        eos_token_id (int, optional): Token ID that terminates generation. Defaults to 151645.
        mm_projector_type (str, optional): Multimodal projector architecture type. Defaults to ``"mlp2x_gelu"``.
        mm_hidden_size (int, optional): Hidden dimensionality of the vision features fed to the projector. Defaults to 3072.
        tokenizer_model_max_length (int, optional): Maximum token sequence length for the tokenizer. Defaults to 8192.
        tokenizer_padding_side (str, optional): Side on which the tokenizer pads sequences. Defaults to ``"right"``.
    """

    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "jingyu"
    ignore_index: int = -100
    image_token_index: int = -200
    eos_token_id: int = 151645
    mm_projector_type: str = "mlp2x_gelu"
    mm_hidden_size: int = 3072
    tokenizer_model_max_length: int = 8192
    tokenizer_padding_side: str = "right"

    @classmethod
    def from_dict(cls, params):
        """Builds a ``ModelConfig`` from a raw config dictionary.

        Flat keys are folded into ``text_config`` when no nested text config exists, and only
        fields declared on this dataclass are forwarded to the constructor.

        Args:
            params (dict): Raw configuration mapping, possibly with nested or flat layouts.

        Returns:
            ModelConfig: The fully constructed model configuration.
        """
        if not params.get("text_config", {}):
            excluded_keys = {"vision_config"}
            params["text_config"] = dict(filter(lambda x: x[0] not in excluded_keys, params.items()))

        if not params.get("vision_config", {}):
            params["vision_config"] = {}

        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})
