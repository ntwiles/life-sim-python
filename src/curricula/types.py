from typing import Callable, Literal

from src.simulation.individual import Individual


CurriculumFn = Callable[[list[Individual]], list[Individual]]
CurriculumKey = Literal['evolutionary', 'reinforcement']
