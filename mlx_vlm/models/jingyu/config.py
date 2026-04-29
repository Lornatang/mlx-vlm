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
        if not params.get("text_config", {}):
            # Copy text config parameters from root level
            excluded_keys = {"vision_config"}
            params["text_config"] = dict(
                filter(lambda x: x[0] not in excluded_keys, params.items())
            )

        if not params.get("vision_config", {}):
            params["vision_config"] = {}

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
