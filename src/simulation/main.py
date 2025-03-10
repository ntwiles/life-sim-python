import math
import random

import tensorflow as tf

from config import LOAD_MODELS, MAX_LENGTH, MUTATION_MAGNITUDE, MUTATION_RATE, NUM_INDIVS, SELECTION_RATE
from src.model.propagation import decide
from src.services.individuals import load_individuals
from src.simulation.heal_zones import HealZone, get_closest_heal_zone, spawn_heal_zones
from src.simulation.individual import Individual, IndividualUpdateContext
from src.utils import normalize_vector

class Simulation:
    generation_time: int
    indivs: list[Individual]
    heal_zones: list[HealZone]

    def __init__(self, indivs: list[Individual]):
        self.generation_time = 0

        self.indivs = indivs
        self.heal_zones = spawn_heal_zones()
        

    def update(self, t: float) -> list[IndividualUpdateContext]:
        return list(map(lambda indiv: self.update_individual(indiv, t), self.indivs))


    def update_individual(self, indiv: Individual, t: float) -> IndividualUpdateContext:
        heal_zone, heal_zone_dist = get_closest_heal_zone(self.heal_zones, indiv.position)

        # TODO: Calculate this in `get_closest_heal_zone`.
        heal_zone_disp = (heal_zone.position[0] - indiv.position[0], heal_zone.position[1] - indiv.position[1])
        heal_zone_dir = normalize_vector(heal_zone_disp)

        if heal_zone_dist < heal_zone.radius:
            indiv.times_healed += 1

        context = IndividualUpdateContext(heal_zone_dir, heal_zone_dist / MAX_LENGTH, indiv.position, indiv.times_healed)

        decision = decide(indiv, context, t)

        indiv.previous_position = indiv.position
        indiv.position = (indiv.position[0] + decision[0], indiv.position[1] + decision[1])

        context.next_position = indiv.position
        return context

def mutate_weights(model: tf.keras.Sequential):
    for var in model.inner.trainable_variables:
        mutation_mask = tf.random.uniform(var.shape) < MUTATION_RATE
        random_mutations = tf.random.normal(var.shape, mean=0.0, stddev=MUTATION_MAGNITUDE)
        var.assign(tf.where(mutation_mask, var + random_mutations, var))

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
        for _ in range(int(1 / SELECTION_RATE)):
            child = Individual()
            child.model.num_simulations = parent.model.num_simulations
            child.model.inner.set_weights(parent.model.inner.get_weights())
            
            mutate_weights(child.model)
            next_generation.append(child)

    return next_generation


