from dataclasses import dataclass

import tensorflow as tf

SIMULATOR_STEPS = 500
NUM_INDIVS = 100
NUM_HEAL_ZONES = 5
HEAL_ZONE_RADIUS = 50
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

class HealZone:
    position: tuple[int, int]
    radius: int

    def __init__(self, position: tuple[int, int], radius: int):
        self.position = position
        self.radius = radius

@dataclass
class IndividualUpdateContext:
    heal_zone_angle: float
    next_position: tuple[int, int]

@dataclass
class PipeMessage:
    indiv_updates: list[IndividualUpdateContext]
    heal_zones: list[HealZone]