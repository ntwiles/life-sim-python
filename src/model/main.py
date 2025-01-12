import tensorflow as tf

from config import INPUT_SIZE

class Model:
    inner: tf.keras.Sequential

    def __init__(self):
        self.inner = tf.keras.Sequential([
            tf.keras.layers.Input(shape=INPUT_SIZE),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
