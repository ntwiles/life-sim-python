from typing import Callable

from src.simulation.individual import Individual

CurriculumFn = Callable[[list[Individual]], list[Individual]]

