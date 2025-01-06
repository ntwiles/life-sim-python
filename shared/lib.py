from dataclasses import dataclass

SIMULATOR_STEPS = 300
NUM_INDIVS = 50
INPUT_SIZE = 5
NUM_HEAL_ZONES = 7
HEAL_ZONE_RADIUS = 30
GRID_SIZE = 300
MAX_LENGTH = GRID_SIZE * 1.414
WINDOW_SCALE = 3
PROFILER = False

MUTATION_RATE = 0.1
MUTATION_MAGNITUDE = 0.01
SELECTION_RATE = 0.25
LOAD_MODELS = True

class HealZone:
    position: tuple[int, int]
    radius: int

    def __init__(self, position: tuple[int, int], radius: int):
        self.position = position
        self.radius = radius

@dataclass
class IndividualUpdateContext:
    heal_zone_dir: tuple[float, float]
    heal_zone_dist: float
    next_position: tuple[int, int]
    times_healed: int

@dataclass
class PipeMessage:
    indiv_updates: list[IndividualUpdateContext]
    heal_zones: list[HealZone]