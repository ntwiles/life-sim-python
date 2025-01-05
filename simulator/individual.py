from random import randint
import tensorflow as tf

from shared.lib import GRID_SIZE, INPUT_SIZE

class Individual:
    position: tuple[int, int]
    previous_position: tuple[int, int]
    times_healed: int
    model: tf.keras.Sequential

    def __init__(self):
        start_position = (randint(0, GRID_SIZE - 1), randint(0, GRID_SIZE - 1))
        self.position = start_position
        self.previous_position = start_position
        self.times_healed = 0

        layers = [
            tf.keras.layers.Input(shape=INPUT_SIZE),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ]

        self.model = tf.keras.Sequential(layers)