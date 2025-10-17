import math
import random

from config import ENABLE_GATING, SELECTION_RATE
from strategies.models import StrategyFn
from model.main import clone_and_mutate_model
from simulation.individual import Individual


def _apply_evolutionary_strategy(generation: list[Individual]) -> list[Individual]:
    breeders = select_breeders(generation)
    return spawn_next_generation(breeders)

apply_evolutionary_strategy: StrategyFn = _apply_evolutionary_strategy


def select_breeders(indivs: list[Individual]) -> list[Individual]:
    min_fitness = min(indiv.times_healed for indiv in indivs)

    baseline = abs(min_fitness) + 1

    adjusted_fitness = [indiv.times_healed + baseline for indiv in indivs]
    total_fitness = sum(adjusted_fitness)

    probabilities = [fitness / total_fitness for fitness in adjusted_fitness]
    num_breeders = math.floor(len(indivs) * SELECTION_RATE)
    
    return random.choices(indivs, weights=probabilities, k=num_breeders)


def spawn_next_generation(breeders: list[Individual]) -> list[Individual]:
    next_generation = []

    for parent in breeders:
        for _ in range(int(round(1 / SELECTION_RATE))):
            child = Individual(clone_and_mutate_model(parent.model, ENABLE_GATING))
            next_generation.append(child)

    return next_generation