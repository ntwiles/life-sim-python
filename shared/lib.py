from dataclasses import dataclass

NUM_INDIVS = 1000
NUM_FOOD = 1000
GRID_SIZE = 500
WINDOW_SCALE = 2

# TODO: Maybe move this to simulator.
@dataclass
class Individual:
    id: int
    position: tuple[int, int]
    previous_position: tuple[int, int]

    def __init__(self, id: int, start_position: tuple[int, int]):
        self.id = id
        self.position = start_position
        self.previous_position = start_position

@dataclass
class IndividualUpdateContext:
    food_angle: float
    next_position: tuple[int, int]

@dataclass
class PipeMessage:
    indiv_updates: list[IndividualUpdateContext]
    food: list[tuple[int, int]]