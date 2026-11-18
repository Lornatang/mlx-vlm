import math
from typing import Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


class LoRaLayer(nn.Module):
    def __init__(
        self,
        linear: Union[nn.Linear, nn.QuantizedLinear],
        rank: int,
        alpha: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_layer = linear

        self.dropout = nn.Dropout(p=dropout)

        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        std_dev = 1 / math.sqrt(rank)

        self.A = mx.random.uniform(
            low=-std_dev,
            high=std_dev,
            shape=(input_dims, rank),
        )
        self.B = mx.zeros((rank, output_dims))
        self.scale = alpha / rank

    def __call__(self, x):
        y = self.original_layer(x)
        lora_update = (self.dropout(x) @ self.A) @ self.B
        return y + (self.scale * lora_update).astype(x.dtype)


def replace_lora_with_linear(model):
    """Fuse legacy LoRaLayer adapters into their base linear layers.

    Adapters are attached at nested module names by get_peft_model, so the
    whole module tree is walked. Each LoRaLayer is replaced by an equivalent
    nn.Linear (or nn.QuantizedLinear, requantized with the original
    group_size, bits and mode) whose forward matches the adapter in eval
    mode.
    """
    replacements = []
    for name, layer in model.named_modules():
        if not isinstance(layer, LoRaLayer):
            continue
        original = layer.original_layer
        weight = original.weight
        is_quantized = isinstance(original, nn.QuantizedLinear)
        if is_quantized:
            weight = mx.dequantize(
                weight,
                original.scales,
                original.biases,
                group_size=original.group_size,
                bits=original.bits,
                mode=original.mode,
            )
        output_dims, input_dims = weight.shape
        use_bias = "bias" in original

        # Create a new Linear layer with the merged parameters
        new_linear_layer = nn.Linear(input_dims, output_dims, bias=use_bias)
        lora_update = (layer.scale * (layer.A @ layer.B)).T.astype(weight.dtype)
        new_linear_layer.weight = weight + lora_update
        if use_bias:
            new_linear_layer.bias = original.bias

        if is_quantized:
            new_linear_layer = nn.QuantizedLinear.from_linear(
                new_linear_layer,
                original.group_size,
                original.bits,
                mode=original.mode,
            )

        replacements.append((name, new_linear_layer))

    if replacements:
        model.update_modules(tree_unflatten(replacements))
    return model
