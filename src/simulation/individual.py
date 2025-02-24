from dataclasses import dataclass
from random import randint

from config import GRID_SIZE
from src.model.main import Model

@dataclass
class IndividualUpdateContext:
    heal_zone_dir: tuple[float, float]
    heal_zone_dist: float
    rad_zone_dir: tuple[float, float]
    rad_zone_dist: float
    next_position: tuple[int, int]
    times_healed: int

class Individual:
    position: tuple[int, int]
    previous_position: tuple[int, int]
    times_healed: int
    model: Model

    def __init__(self):
        start_position = (randint(0, GRID_SIZE - 1), randint(0, GRID_SIZE - 1))
        self.position = start_position
        self.previous_position = start_position
        self.times_healed = 0

        self.model = Model()