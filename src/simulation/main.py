import math
import random

import tensorflow as tf

from config import LOAD_MODELS, NUM_INDIVS, SELECTION_RATE
from src.model.main import mutate_weights_with_gating
from src.simulation.rad_zones import RadZone, spawn_rad_zones
from src.services.individuals import load_individuals
from src.simulation.heal_zones import HealZone, spawn_heal_zones
from src.simulation.individual import Individual, IndividualUpdateContext

class Simulation:
    generation_time: int
    indivs: list[Individual]
    heal_zones: list[HealZone]
    rad_zones: list[RadZone]

    def __init__(self, indivs: list[Individual]):
        self.generation_time = 0

        self.indivs = indivs
        self.heal_zones = spawn_heal_zones()
        self.rad_zones = spawn_rad_zones()

    def update(self, t: float) -> list[IndividualUpdateContext]:
        for rad_zone in self.rad_zones:
            rad_zone.update()

        return list(map(lambda indiv: indiv.update(self.heal_zones, self.rad_zones), self.indivs))



def select_breeders(indivs: list[Individual]) -> list[Individual]:
    total_fitness = sum(indiv.times_healed for indiv in indivs)

    if total_fitness == 0:
        probabilities = [1 / len(indivs)] * len(indivs)
    else:
        probabilities = [indiv.times_healed / total_fitness for indiv in indivs]

    num_breeders = math.floor(len(indivs) * SELECTION_RATE)
    
    return random.choices(indivs, weights=probabilities, k=num_breeders)

def spawn_initial_generation() -> list[Individual]:
    if LOAD_MODELS:
        return load_individuals()
    else:
        return [Individual() for _ in range(NUM_INDIVS)]


def spawn_next_generation(breeders: list[Individual]) -> list[Individual]:
    next_generation = []

    for parent in breeders:
        for _ in range(int(round(1 / SELECTION_RATE))):
            child = Individual()
            child.model.num_simulations = parent.model.num_simulations
            child.model.inner.set_weights(parent.model.inner.get_weights())
            
            mutate_weights_with_gating(child.model)
            next_generation.append(child)

    return next_generation


