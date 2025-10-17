from typing import Any
from numpy.typing import NDArray
import tensorflow as tf

from model.main import Model

def batch_decide(models: list[Model], inputs: NDArray[Any]) -> list[tuple[int, int]]:
    output_values = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (0, 0),
    ]
     
    batch_inputs = tf.constant(inputs, dtype=tf.float32)

    ref_model = models[0]

    x = batch_inputs

    gate_idx = 0
    for layer in ref_model.inner.layers:
        if hasattr(layer, 'kernel'):  # Dense layer with weights
            all_gate_masks = tf.stack([m.gate_masks[gate_idx] for m in models])

            gate_idx += 1

            layer_weights = tf.expand_dims(layer.kernel, axis=0) 
            gated_weights = layer_weights * all_gate_masks  

            x = tf.einsum('bi,bij->bj', x, gated_weights)

            if hasattr(layer, 'bias') and layer.use_bias:
                all_bias_gate_masks = tf.stack([m.gate_masks[gate_idx] for m in models])
                gate_idx += 1

                layer_biases = tf.expand_dims(layer.bias, axis=0)
                gated_biases = layer_biases * all_bias_gate_masks
                x = x + gated_biases

            if layer.activation is not None:
                x = layer.activation(x)

        else:
            # For Input layers or layers without weights, just pass through
            x = layer(x)

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

    # Apply gate masks during forward pass
    x = tf.constant(input, dtype=tf.float32)
    
    gate_idx = 0
    for layer in model.inner.layers:
        if hasattr(layer, 'kernel'):  # Dense layer with weights
            # Apply gate mask to weights
            gated_kernel = layer.kernel * model.gate_masks[gate_idx]
            gate_idx += 1
            
            # Manual matrix multiplication
            x = tf.matmul(x, gated_kernel)
            
            # Apply bias if it exists
            if hasattr(layer, 'bias') and layer.use_bias:
                gated_bias = layer.bias * model.gate_masks[gate_idx]
                gate_idx += 1
                x = x + gated_bias
            
            # Apply activation function
            if layer.activation is not None:
                x = layer.activation(x)
        else:
            # For Input layers or layers without weights, just pass through
            x = layer(x)
    
    output = x
    decision = tf.argmax(output, axis=1).numpy()[0]

    return output_values[decision]