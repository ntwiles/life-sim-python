from dataclasses import dataclass
from random import randint
import numpy as np

from config import GRID_SIZE, INPUT_SIZE, MAX_LENGTH
from src.model.propagation import decide
from src.simulation.heal_zones import HealZone, get_closest_heal_zone
from src.simulation.rad_zones import RadZone, get_closest_rad_zone
from src.model.main import Model
from src.utils import normalize_vector

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

    def handle_heal_zones(self, heal_zones: list[HealZone]):
        heal_zone, heal_zone_dist = get_closest_heal_zone(heal_zones, self.position)

        # TODO: Calculate this in `get_closest_heal_zone`.
        heal_zone_disp = (heal_zone.position[0] - self.position[0], heal_zone.position[1] - self.position[1])
        heal_zone_dir = normalize_vector(heal_zone_disp)

        if heal_zone_dist < heal_zone.radius:
            self.times_healed += 1

        return (heal_zone_dir, heal_zone_dist)
    
    def handle_rad_zones(self, rad_zones: list[RadZone]):
        rad_zone, rad_zone_dist = get_closest_rad_zone(rad_zones, self.position)
        
        # TODO: Calculate this in `get_closest_rad_zone`.
        rad_zone_disp = (rad_zone.position[0] - self.position[0], rad_zone.position[1] - self.position[1])
        rad_zone_dir = normalize_vector(rad_zone_disp)

        if rad_zone_dist < rad_zone.radius:
            self.times_healed -= 1

        return (rad_zone_dir, rad_zone_dist)

    def update(self, heal_zones: list[HealZone], rad_zones: list[RadZone]) -> IndividualUpdateContext:
        (heal_zone_dir, heal_zone_dist) = self.handle_heal_zones(heal_zones)
        (rad_zone_dir, rad_zone_dist) = self.handle_rad_zones(rad_zones)

        context = IndividualUpdateContext(
            heal_zone_dir, 
            heal_zone_dist / MAX_LENGTH,
            rad_zone_dir,
            rad_zone_dist / MAX_LENGTH,
            self.position, 
            self.times_healed
        )

        decision = decide(self.model, np.array([self.calculate_input_values(context)]))

        self.previous_position = self.position
        self.position = (self.position[0] + decision[0], self.position[1] + decision[1])

        context.next_position = self.position
        return context
    
    def calculate_input_values(self, context: IndividualUpdateContext) -> list[float]:
        prev_position_dir_x = self.position[0] - self.previous_position[0]
        prev_position_dir_y = self.position[1] - self.previous_position[1]

        heal_zone_dir_x, heal_zone_dir_y = context.heal_zone_dir
        rad_zone_dir_x, rad_zone_dir_y = context.rad_zone_dir

        input_values = [
            prev_position_dir_x,
            prev_position_dir_y,
            context.heal_zone_dist,
            heal_zone_dir_x,
            heal_zone_dir_y,
            context.rad_zone_dist,
            rad_zone_dir_x,
            rad_zone_dir_y,
        ]

        if len(input_values) != INPUT_SIZE:
            raise ValueError(f"Expected {INPUT_SIZE} inputs, got {len(input_values)}")
        
        return input_values