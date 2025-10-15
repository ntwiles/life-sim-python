from typing import Callable, Literal

from src.simulation.individual import Individual

StrategyFn = Callable[[list[Individual]], list[Individual]]
StrategyKey = Literal['evolutionary', 'reinforcement']
