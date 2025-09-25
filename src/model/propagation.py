from typing import Any
from numpy.typing import NDArray
import tensorflow as tf

from src.model.main import Model

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