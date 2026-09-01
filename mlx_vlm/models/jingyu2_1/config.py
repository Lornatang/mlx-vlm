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
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    """Hyperparameters for the language-model (LFM2) backbone.

    Attributes:
        model_type (str, optional): Identifier for the language-model architecture. Defaults to ``"lfm2"``.
        hidden_size (int, optional): Dimensionality of the LLM embeddings. Defaults to 1024.
        num_hidden_layers (int, optional): Number of transformer layers in the LLM. Defaults to 16.
        intermediate_size (int, optional): Size of the feed-forward inner projection. Defaults to 6656.
        num_attention_heads (int, optional): Number of query heads in self-attention. Defaults to 16.
        num_key_value_heads (int, optional): Number of key/value heads for grouped-query attention. Defaults to 8.
        max_position_embeddings (int, optional): Maximum sequence length the position encodings support.
            Defaults to 128000.
        rope_theta (float, optional): Base frequency for rotary position embeddings (RoPE). Defaults to 1000000.0.
        vocab_size (int, optional): Size of the token vocabulary. Defaults to 65536.
        eos_token_id (int, optional): Token ID that terminates generation. Defaults to 7.
        initializer_range (float, optional): Standard deviation used for weight initialization. Defaults to 0.02.
        norm_eps (float, optional): Epsilon used in normalization layers for numerical stability. Defaults to 1e-05.
        use_cache (bool, optional): Whether the model returns the last hidden state for KV caching.
            Defaults to ``True``.
        use_pos_enc (bool, optional): Whether positional encodings are applied. Defaults to ``True``.
        block_auto_adjust_ff_dim (bool, optional): Whether to automatically adjust the block feed-forward
            dimension to satisfy the alignment constraint. Defaults to ``True``.
        block_dim (int, optional): Channel dimensionality of each convolutional block. Defaults to 1024.
        block_ff_dim (int, optional): Size of the inner feed-forward projection of each block. Defaults to 6656.
        block_ffn_dim_multiplier (float, optional): Multiplier scaling the block feed-forward size.
            Defaults to 1.0.
        block_mlp_init_scale (float, optional): Scale applied to the MLP weights at initialization.
            Defaults to 1.0.
        block_multiple_of (int, optional): Alignment constraint applied to the feed-forward inner size.
            Defaults to 256.
        block_norm_eps (float, optional): Epsilon used in the block normalization layers. Defaults to 1e-05.
        block_out_init_scale (float, optional): Scale applied to the output projection weights at
            initialization. Defaults to 1.0.
        block_use_swiglu (bool, optional): Whether the feed-forward layers use the SwiGLU activation.
            Defaults to ``True``.
        block_use_xavier_init (bool, optional): Whether the block weights use Xavier initialization.
            Defaults to ``True``.
        conv_L_cache (int, optional): Kernel length of the short convolution layer. Defaults to 3.
        conv_bias (bool, optional): Whether the convolution layers include a bias term. Defaults to ``False``.
        conv_dim (int, optional): Channel dimensionality of the convolution layer. Defaults to 1024.
        conv_dim_out (int, optional): Output channel dimensionality of the convolution layer. Defaults to 1024.
        conv_use_xavier_init (bool, optional): Whether the convolution weights use Xavier initialization.
            Defaults to ``True``.
        layer_types (Optional[List[str]], optional): Type of each layer (e.g. ``"full_attention"``).
            Defaults to ``None``.
        num_heads (int, optional): Number of attention heads in the full-attention layers. Defaults to 16.
        full_attn_idxs (Optional[List[int]], optional): Indices of layers that use full attention; derived
            from ``layer_types`` in ``__post_init__``. Defaults to ``None``.
    """

    model_type: str = "lfm2"
    hidden_size: int = 1024
    num_hidden_layers: int = 16
    intermediate_size: int = 6656
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0
    vocab_size: int = 65536
    eos_token_id: int = 7
    initializer_range: float = 0.02
    norm_eps: float = 1e-05
    use_cache: bool = True
    use_pos_enc: bool = True
    block_auto_adjust_ff_dim: bool = True
    block_dim: int = 1024
    block_ff_dim: int = 6656
    block_ffn_dim_multiplier: float = 1.0
    block_mlp_init_scale: float = 1.0
    block_multiple_of: int = 256
    block_norm_eps: float = 1e-05
    block_out_init_scale: float = 1.0
    block_use_swiglu: bool = True
    block_use_xavier_init: bool = True
    conv_L_cache: int = 3
    conv_bias: bool = False
    conv_dim: int = 1024
    conv_dim_out: int = 1024
    conv_use_xavier_init: bool = True
    layer_types: List[str] = None
    num_heads: int = 16
    full_attn_idxs: List[int] = None

    @classmethod
    def from_dict(cls, params):
        """Builds a ``TextConfig`` from a raw config dictionary.

        Before dispatching to :meth:`BaseModelConfig.from_dict`, ``block_ff_dim`` is back-filled
        with the ``intermediate_size`` value when it is not explicitly provided.

        Args:
            params (dict): Raw configuration mapping.

        Returns:
            TextConfig: The fully constructed language-model configuration.
        """
        params = dict(params)
        params.setdefault("block_ff_dim", params.get("intermediate_size", 6656))
        return super().from_dict(params)

    def __post_init__(self):
        """Derives ``full_attn_idxs`` from ``layer_types``.

        Scans the layer-type sequence and records the indices whose value is ``"full_attention"``.
        """
        if self.full_attn_idxs is None:
            self.full_attn_idxs = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]


@dataclass
class VisionConfig(BaseModelConfig):
    """Hyperparameters for the FastViTHD vision tower.

    Attributes:
        model_type (str, optional): Identifier for the vision encoder architecture. Defaults to ``"jingyu2_1"``.
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

    model_type: str = "jingyu2_1"
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
        model_type (str, optional): Identifier for the merged architecture. Defaults to ``"jingyu2_1"``.
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
    model_type: str = "jingyu2_1"
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
