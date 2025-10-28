
from dataclasses import dataclass

from simulation.heal_zones import HealZone
from simulation.individual import IndividualUpdateContext
from simulation.rad_zones import RadZone


@dataclass
class SimulationDrawingData:
    indiv_updates: list[IndividualUpdateContext]
    heal_zones: list[HealZone]
    rad_zones: list[RadZone]
    steps_remaining: int

@dataclass
class ProjectDrawingData:
    last_sim_duration: float
    last_training_duration: float
    avg_times_healed: float
    generation_num: int