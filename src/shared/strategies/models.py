from typing import Callable, Literal

from trainer.simulation.individual import Individual

StrategyFn = Callable[[list[Individual]], list[Individual]]
StrategyKey = Literal['evolutionary', 'reinforcement']
