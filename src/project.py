from collections import deque
from collections.abc import Callable
import time

from config import LOAD_MODELS, NUM_INDIVS, SIMULATOR_STEPS
from src.curricula.main import CurriculumFn
from src.drawing_data import SimulationDrawingData, ProjectDrawingData
from src.simulation.individual import Individual
from src.fitness import calculate_theoretical_max_fitness
from src.services.individuals import load_individuals, save_individuals
from src.simulation.main import Simulation

class Project:
    sim: Simulation | None
    apply_curriculum: CurriculumFn

    last_k_avg_times_healed: deque[float]
    theoretical_max_fitness: float

    def __init__(self, apply_curriculum: CurriculumFn):
        self.last_k_avg_times_healed = deque(maxlen=20)
        self.theoretical_max_fitness = calculate_theoretical_max_fitness()
        self.sim = None
        self.apply_curriculum = apply_curriculum

    def run(self, on_sim_update: Callable[[SimulationDrawingData], None] | None = None, on_project_update: Callable[[ProjectDrawingData], None] | None = None):
        generation = spawn_initial_generation()
        running_curriculum = True

        while running_curriculum:
            sim_time_started = time.time()

            self.sim = Simulation(generation, on_update=on_sim_update)
            self.sim.run(SIMULATOR_STEPS)

            sim_duration = time.time() - sim_time_started

            training_time_started = time.time() 

            avg_times_healed = sum(map(lambda indiv: indiv.times_healed, generation)) / len(generation)

            for indiv in generation:
                indiv.model.num_generations += 1

            self.last_k_avg_times_healed.append(avg_times_healed)
            moving_avg_times_healed = sum(self.last_k_avg_times_healed) / len(self.last_k_avg_times_healed)

            if moving_avg_times_healed > self.theoretical_max_fitness * .8:
                # We've hit 80% of the theoretical max fitness, so we can stop now.
                save_individuals(generation)
                running_curriculum = False
            elif indiv.model.num_generations % 100 == 0:
                save_individuals(generation)

            generation = self.apply_curriculum(generation)

            if on_project_update is not None:
                on_project_update(ProjectDrawingData(
                    last_sim_duration=sim_duration,
                    last_training_duration=time.time() - training_time_started,
                    moving_avg_times_healed=moving_avg_times_healed,
                    avg_times_healed=avg_times_healed
                ))


def spawn_initial_generation() -> list[Individual]:
    if LOAD_MODELS:
        return load_individuals()
    else:
        return [Individual() for _ in range(NUM_INDIVS)]

