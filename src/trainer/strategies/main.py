from trainer.strategies.evolutionary import apply_evolutionary_strategy
from trainer.strategies.reinforcement import apply_reinforcement_strategy
from trainer.strategies.types import StrategyFn

strategy_functions: dict[str, StrategyFn] = {
    "evolutionary": apply_evolutionary_strategy, 
    "reinforcement": apply_reinforcement_strategy, 
}

