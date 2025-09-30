from collections import deque
from collections.abc import Callable
import math
import random
import time

from config import ENABLE_GATING, LOAD_MODELS, NUM_INDIVS, SELECTION_RATE, SIMULATOR_STEPS
from src.drawing_data import DrawingData
from src.model.main import clone_and_mutate_model
from src.simulation.individual import Individual
from src.fitness import calculate_theoretical_max_fitness
from src.services.individuals import load_individuals, save_individuals
from src.simulation.main import Simulation

class Curriculum:
    sim: Simulation | None

    avg_times_healed: float
    last_k_avg_times_healed: deque[float]
    moving_avg_times_healed: float
    theoretical_max_fitness: float

    time_started: float
    last_run_time: float


    def __init__(self):
        self.avg_times_healed = 0
        self.moving_avg_times_healed = 0
        self.last_k_avg_times_healed = deque(maxlen=20)

        self.theoretical_max_fitness = calculate_theoretical_max_fitness()

        self.sim = None

        self.last_run_time = 0.0
        self.time_started = time.time()


    def run(self, on_sim_update: Callable[[DrawingData], None] | None = None):
        generation = spawn_initial_generation()
        running_curriculum = True

        while running_curriculum:
            self.sim = Simulation(generation, on_update=on_sim_update)
            self.sim.run(SIMULATOR_STEPS)

            self.avg_times_healed = sum(map(lambda indiv: indiv.times_healed, generation)) / len(generation)

            for indiv in generation:
                indiv.model.num_simulations += 1

            save_individuals(generation)

            breeders = select_breeders(generation)
            generation = spawn_next_generation(breeders)
            
            # TODO: Maybe this should be done in the Simulation class.
            self.last_k_avg_times_healed.append(self.avg_times_healed)
            self.moving_avg_times_healed = sum(self.last_k_avg_times_healed) / len(self.last_k_avg_times_healed)
            self.last_run_time = time.time() - self.time_started
            self.time_started = time.time()

            # We've hit 80% of the theoretical max fitness, so we can stop now.
            if self.moving_avg_times_healed > self.theoretical_max_fitness * .8:
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