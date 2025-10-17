from strategies.evolutionary import apply_evolutionary_strategy
from strategies.reinforcement import apply_reinforcement_strategy
from strategies.models import StrategyFn

strategy_functions: dict[str, StrategyFn] = {
    "evolutionary": apply_evolutionary_strategy, 
    "reinforcement": apply_reinforcement_strategy, 
}

