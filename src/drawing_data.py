
from dataclasses import dataclass

from src.simulation.heal_zones import HealZone
from src.simulation.individual import IndividualUpdateContext
from src.simulation.rad_zones import RadZone


@dataclass
class DrawingData:
    indiv_updates: list[IndividualUpdateContext]
    heal_zones: list[HealZone]
    rad_zones: list[RadZone]
    steps_remaining: int
    model_num_generations: int