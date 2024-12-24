from dataclasses import dataclass

NUM_INDIVS = 1000
GRID_SIZE = 500
WINDOW_SCALE = 2

@dataclass
class Individual:
    id: int
    position: tuple[int, int]
