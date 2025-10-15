import numpy as np

from src.strategies.types import StrategyFn
from src.simulation.individual import Individual


def _apply_reinforcement_strategy(generation: list[Individual]) -> list[Individual]:
    returns = []
    
    for indiv in generation:
        if hasattr(indiv, "episode_reward_deltas") and indiv.episode_reward_deltas:
            R = float(np.sum(indiv.episode_reward_deltas))
        else:
            R = float(indiv.times_healed)
        returns.append(R)
    if not returns:
        return generation
    

apply_reinforcement_strategy: StrategyFn = _apply_reinforcement_strategy