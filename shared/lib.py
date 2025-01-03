from dataclasses import dataclass

import tensorflow as tf

SIMULATOR_RUNS = 100000
SIMULATOR_STEPS = 200
NUM_INDIVS = 50
INPUT_SIZE = 6
NUM_HEAL_ZONES = 5
HEAL_ZONE_RADIUS = 50
GRID_SIZE = 500
MAX_LENGTH = GRID_SIZE * 1.414
WINDOW_SCALE = 2
PROFILER = False

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
        self.model.build((None, INPUT_SIZE))

class HealZone:
    position: tuple[int, int]
    radius: int

    def __init__(self, position: tuple[int, int], radius: int):
        self.position = position
        self.radius = radius

@dataclass
class IndividualUpdateContext:
    heal_zone_angle: float
    heal_zone_dist: float
    next_position: tuple[int, int]
    times_healed: int

@dataclass
class PipeMessage:
    indiv_updates: list[IndividualUpdateContext]
    heal_zones: list[HealZone]