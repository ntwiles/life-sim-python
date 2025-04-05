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

    output = model.inner(input)
    decision = tf.argmax(output, axis=1).numpy()[0]

    return output_values[decision]