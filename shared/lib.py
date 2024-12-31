from dataclasses import dataclass

import tensorflow as tf

SIMULATOR_STEPS = 500
NUM_INDIVS = 100
NUM_FOOD = 1000
GRID_SIZE = 500
WINDOW_SCALE = 2

# TODO: Maybe move this to simulator.
class Individual:
    id: int
    position: tuple[int, int]
    previous_position: tuple[int, int]
    model: tf.keras.Sequential

    def __init__(self, id: int, start_position: tuple[int, int]):
        self.id = id
        self.position = start_position
        self.previous_position = start_position

        layers = [
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ]

        self.model = tf.keras.Sequential(layers)

@dataclass
class IndividualUpdateContext:
    food_angle: float
    next_position: tuple[int, int]

@dataclass
class PipeMessage:
    indiv_updates: list[IndividualUpdateContext]
    food: list[tuple[int, int]]