from dataclasses import dataclass

import tensorflow as tf

SIMULATOR_STEPS = 500
NUM_INDIVS = 100
INPUT_SIZE = 5
NUM_HEAL_ZONES = 5
HEAL_ZONE_RADIUS = 50
GRID_SIZE = 500
WINDOW_SCALE = 2

# TODO: Maybe move this to simulator.
class Individual:
    position: tuple[int, int]
    previous_position: tuple[int, int]
    times_healed: int
    model: tf.keras.Sequential

    def __init__(self, start_position: tuple[int, int]):
        self.position = start_position
        self.previous_position = start_position
        self.times_healed = 0

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