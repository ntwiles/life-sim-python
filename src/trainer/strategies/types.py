
from typing import Callable

from trainer.simulation.individual import Individual

StrategyFn = Callable[[list[Individual]], list[Individual]]
