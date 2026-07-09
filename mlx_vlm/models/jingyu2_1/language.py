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
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..cache import ArraysCache, KVCache
from ..lfm2.language import Lfm2Model
from .config import TextConfig


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2Model(config)

    def __call__(
        self,
        inputs: Optional[mx.array],
        mask: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs_embeds is None:
            inputs_embeds = kwargs.pop("input_embeddings", None)

        if inputs is None and inputs_embeds is None:
            raise ValueError("Either `inputs` or `inputs_embeds` must be provided.")

        out = self.model(inputs, cache, inputs_embeds)
        out = self.model.embed_tokens.as_linear(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        weights = dict(weights)
        if self.config.tie_word_embeddings:
            weights.pop("language_model.lm_head.weight", None)
            weights.pop("lm_head.weight", None)

        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)

            sanitized_weights[name] = param
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() if l.is_attention_layer else ArraysCache(size=1) for l in self.layers]
