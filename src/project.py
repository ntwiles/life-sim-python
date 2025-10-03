from collections import deque
from collections.abc import Callable
import math
import random
import time

from config import ENABLE_GATING, LOAD_MODELS, NUM_INDIVS, SELECTION_RATE, SIMULATOR_STEPS
from src.drawing_data import SimulationDrawingData, ProjectDrawingData
from src.model.main import clone_and_mutate_model
from src.simulation.individual import Individual
from src.fitness import calculate_theoretical_max_fitness
from src.services.individuals import load_individuals, save_individuals
from src.simulation.main import Simulation

class Project:
    sim: Simulation | None

    avg_times_healed: float
    last_k_avg_times_healed: deque[float]
    moving_avg_times_healed: float
    theoretical_max_fitness: float

    def __init__(self):
        self.avg_times_healed = 0
        self.moving_avg_times_healed = 0
        self.last_k_avg_times_healed = deque(maxlen=20)

        self.theoretical_max_fitness = calculate_theoretical_max_fitness()

        self.sim = None

    def run(self, on_sim_update: Callable[[SimulationDrawingData], None] | None = None, on_project_update: Callable[[ProjectDrawingData], None] | None = None):
        generation = spawn_initial_generation()
        running_curriculum = True

        while running_curriculum:
            sim_time_started = time.time()

            self.sim = Simulation(generation, on_update=on_sim_update)
            self.sim.run(SIMULATOR_STEPS)

            sim_duration = time.time() - sim_time_started

            training_time_started = time.time() 

            self.avg_times_healed = sum(map(lambda indiv: indiv.times_healed, generation)) / len(generation)

            for indiv in generation:
                indiv.model.num_generations += 1

            if indiv.model.num_generations % 100 == 0:
                save_individuals(generation)

            breeders = select_breeders(generation)
            generation = spawn_next_generation(breeders)
            
            # TODO: Maybe this should be done in the Simulation class.
            self.last_k_avg_times_healed.append(self.avg_times_healed)
            self.moving_avg_times_healed = sum(self.last_k_avg_times_healed) / len(self.last_k_avg_times_healed)

            training_duration = time.time() - training_time_started

            if on_project_update is not None:
                on_project_update(ProjectDrawingData(
                    last_sim_duration=sim_duration,
                    last_training_duration=training_duration
                ))

            # We've hit 80% of the theoretical max fitness, so we can stop now.
            if self.moving_avg_times_healed > self.theoretical_max_fitness * .8:
                save_individuals(generation)
                running_curriculum = False


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
            child = Individual(clone_and_mutate_model(parent.model, ENABLE_GATING))
            next_generation.append(child)

    return next_generation

