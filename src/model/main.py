from dataclasses import dataclass
import tensorflow as tf

from config import INPUT_SIZE

@dataclass
class Model:
    inner: tf.keras.Sequential
    num_simulations: int


def create_model() -> Model:
    inner = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(INPUT_SIZE,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
    
    return Model(inner=inner, num_simulations=0)
1