
from dataclasses import dataclass

from shared.types.individual_update_context import IndividualUpdateContext

@dataclass
class HealZoneDrawingData:
    position: tuple[int, int]
    radius: int

@dataclass 
class RadZoneDrawingData:
    position: tuple[int, int]
    radius: int


@dataclass
class SimulationDrawingData:
    indiv_updates: list[IndividualUpdateContext]
    heal_zones: list[HealZoneDrawingData]
    rad_zones: list[RadZoneDrawingData]
    steps_remaining: int

@dataclass
class ProjectDrawingData:
    last_sim_duration: float
    last_training_duration: float
    avg_times_healed: float
    generation_num: int