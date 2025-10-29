from collections import deque
from collections.abc import Callable
import uuid
import time

from shared.config import LOAD_MODELS, NUM_INDIVS, PLOT_MAX_DATA_POINTS, SIM_STEPS
from shared.types.strategies import StrategyKey
from trainer.strategies.main import StrategyFn, strategy_functions
from shared.types.drawing_data import SimulationDrawingData, ProjectDrawingData
from trainer.simulation.individual import Individual
from trainer.simulation.fitness import calculate_theoretical_max_fitness
from shared.services.individuals import load_individual_data, save_individual_data
from shared.services.projects import ProjectData, save_project_data
from trainer.simulation.main import Simulation

class Project:
    # TODO: Do we need to store sim here?
    sim: Simulation | None

    strategy_key: StrategyKey
    apply_strategy: StrategyFn
    id: uuid.UUID
    last_k_avg_fitness: deque[float]
    theoretical_max_fitness: float
    generation_num: int

    def __init__(self):
        self.last_k_avg_fitness = deque(maxlen=PLOT_MAX_DATA_POINTS)
        self.theoretical_max_fitness = calculate_theoretical_max_fitness()
        self.sim = None

    @staticmethod
    def from_data(project_data: ProjectData) -> "Project":
        project = Project()
        project.last_k_avg_fitness = project_data.last_k_avg_fitness
        project.theoretical_max_fitness = calculate_theoretical_max_fitness()
        project.sim = None
        project.strategy_key = project_data.strategy
        project.apply_strategy = strategy_functions[project.strategy_key]
        project.id = project_data.id
        project.generation_num = project_data.generation_num
        return project
    
    def to_data(self) -> ProjectData:
        return ProjectData(
            id=self.id,
            last_k_avg_fitness=self.last_k_avg_fitness,
            strategy=self.strategy_key,
            generation_num=self.generation_num
        )

    @staticmethod
    def new(strategy_key: StrategyKey) -> "Project":
        project = Project()
        project.last_k_avg_fitness = deque(maxlen=20)
        project.theoretical_max_fitness = calculate_theoretical_max_fitness()
        project.sim = None
        project.strategy_key = strategy_key
        project.apply_strategy = strategy_functions[project.strategy_key]
        project.id = uuid.uuid4()
        return project

    def run(self, on_sim_update: Callable[[SimulationDrawingData], None] | None = None, on_project_update: Callable[[ProjectDrawingData], None] | None = None):
        generation = self.spawn_initial_generation()
        running_strategy = True

        while running_strategy:
            sim_time_started = time.time()

            self.sim = Simulation(generation, on_update=on_sim_update)
            self.sim.run(SIM_STEPS)

            sim_duration = time.time() - sim_time_started

            training_time_started = time.time() 

            avg_times_healed = sum(map(lambda indiv: indiv.times_healed, generation)) / len(generation)

            self.last_k_avg_fitness.append(avg_times_healed)
            moving_avg_times_healed = sum(self.last_k_avg_fitness) / len(self.last_k_avg_fitness)

            if moving_avg_times_healed > self.theoretical_max_fitness * .8:
                # We've hit 80% of the theoretical max fitness, so we can stop now.
                save_project_data(self.to_data())

                for idx, indiv in enumerate(generation):
                    save_individual_data(self.id, idx, indiv.to_data())

                running_strategy = False
            elif self.generation_num % 10 == 0:
                save_project_data(self.to_data())

                for idx, indiv in enumerate(generation):
                    save_individual_data(self.id, idx, indiv.to_data())

            generation = self.apply_strategy(generation)

            self.generation_num += 1

            if on_project_update is not None:
                on_project_update(ProjectDrawingData(
                    last_sim_duration=sim_duration,
                    last_training_duration=time.time() - training_time_started,
                    avg_times_healed=avg_times_healed,
                    generation_num=self.generation_num
                ))


    def spawn_initial_generation(self) -> list[Individual]:
        if LOAD_MODELS:
            return [Individual.from_data(load_individual_data(self.id, i)) for i in range(NUM_INDIVS)]

        else:
            return [Individual() for _ in range(NUM_INDIVS)]

