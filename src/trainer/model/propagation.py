from typing import Any
from numpy.typing import NDArray
import tensorflow as tf

from trainer.model.main import Model

def _make_forward_fn(ref_model: Model):
    """Build a jit-compiled forward function closed over the layer structure of ref_model."""
    dense_layers = [l for l in ref_model.inner.layers if hasattr(l, 'kernel')]

    @tf.function()
    def forward(x):
        for layer in dense_layers:
            x = tf.matmul(x, layer.kernel)
            if hasattr(layer, 'bias') and layer.use_bias:
                x = x + layer.bias
            if layer.activation is not None:
                x = layer.activation(x)
        return x

    return forward


_forward_fn = None


def batch_decide(models: list[Model], inputs: NDArray[Any]) -> list[tuple[int, int]]:
    global _forward_fn

    output_values = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (0, 0),
    ]

    batch_inputs = tf.constant(inputs, dtype=tf.float32)

    ref_model = models[0]

    if _forward_fn is None:
        _forward_fn = _make_forward_fn(ref_model)

    x = _forward_fn(batch_inputs)

    decisions = tf.argmax(x, axis=1).numpy()
    return [output_values[decision] for decision in decisions]


def decide(model: Model, input: NDArray[Any]) -> tuple[int, int]:
    output_values = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (0, 0),
    ]

    x = tf.constant(input, dtype=tf.float32)

    for layer in model.inner.layers:
        if hasattr(layer, 'kernel'):
            x = tf.matmul(x, layer.kernel)
            if hasattr(layer, 'bias') and layer.use_bias:
                x = x + layer.bias
            if layer.activation is not None:
                x = layer.activation(x)
        else:
            x = layer(x)

    return output_values[tf.argmax(x, axis=1).numpy()[0]]