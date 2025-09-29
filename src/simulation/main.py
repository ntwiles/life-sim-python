import math
import random

import numpy as np
import tensorflow as tf

from config import GRID_SIZE, LOAD_MODELS, NUM_INDIVS, SELECTION_RATE
from src.model.propagation import batch_decide
from src.model.main import clone_model
from src.simulation.rad_zones import RadZone, spawn_rad_zones
from src.services.individuals import load_individuals
from src.simulation.heal_zones import HealZone, spawn_heal_zones
from src.simulation.individual import Individual, IndividualUpdateContext

class Simulation:
    indivs: list[Individual]
    heal_zones: list[HealZone]
    rad_zones: list[RadZone]

    indiv_updates: list[IndividualUpdateContext] | None
    steps_remaining: int

    def __init__(self, indivs: list[Individual]):
        self.indivs = indivs
        self.heal_zones = spawn_heal_zones()
        self.rad_zones = spawn_rad_zones()
        self.indiv_updates = None
        self.steps_remaining = 0

    def run(self, num_steps: int):
        self.steps_remaining = num_steps
        while self.steps_remaining > 0:
            self.update()
            self.steps_remaining -= 1


    # TODO: Get a better understanding of this `with` usage here. Do we actually need the GPU for all this logic?
    def update(self):
        with tf.device('/GPU:0'):
            for rad_zone in self.rad_zones:
                rad_zone.update()

            contexts = []
            all_inputs = []

            for indiv in self.indivs:
                context = indiv.update(self.heal_zones, self.rad_zones)
                all_inputs.append(indiv.calculate_input_values(context))
                contexts.append(context)

            all_decisions = batch_decide([indiv.model for indiv in self.indivs], np.array(all_inputs))

            for indiv, decision, context in zip(self.indivs, all_decisions, contexts):
                indiv.previous_position = indiv.position
                new_position = (indiv.position[0] + decision[0], indiv.position[1] + decision[1])

                indiv.position = (
                    max(0, min(new_position[0], GRID_SIZE - 1)),
                    max(0, min(new_position[1], GRID_SIZE - 1))
                )

                context.next_position = indiv.position 

            self.indiv_updates = contexts

def select_breeders(indivs: list[Individual]) -> list[Individual]:
    min_fitness = min(indiv.times_healed for indiv in indivs)

    baseline = abs(min_fitness) + 1

    adjusted_fitness = [indiv.times_healed + baseline for indiv in indivs]
    total_fitness = sum(adjusted_fitness)

    probabilities = [fitness / total_fitness for fitness in adjusted_fitness]
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
            child = Individual(clone_model(parent.model))
            next_generation.append(child)

    return next_generation


