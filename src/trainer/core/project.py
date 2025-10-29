from collections import deque
import uuid

from shared.config import  PLOT_MAX_DATA_POINTS
from shared.types.strategies import StrategyKey
from shared.types.drawing_data import  ProjectDrawingData
from trainer.core.simulation_orchestrator import GenerationCompleteData
from trainer.simulation.fitness import calculate_theoretical_max_fitness
from shared.services.individuals import save_individual_data
from shared.services.projects import ProjectData, save_project_data

class Project:
    strategy_key: StrategyKey
    id: uuid.UUID
    last_k_avg_fitness: deque[float]
    theoretical_max_fitness: float
    generation_num: int

    def __init__(self):
        self.last_k_avg_fitness = deque(maxlen=PLOT_MAX_DATA_POINTS)
        self.theoretical_max_fitness = calculate_theoretical_max_fitness()

    @staticmethod
    def from_data(project_data: ProjectData) -> "Project":
        project = Project()
        project.last_k_avg_fitness = project_data.last_k_avg_fitness
        project.theoretical_max_fitness = calculate_theoretical_max_fitness()
        project.strategy_key = project_data.strategy
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
    
    def update_project(self, data: GenerationCompleteData) -> tuple[ProjectDrawingData, bool]:
        is_training_complete = False
        self.generation_num += 1

        avg_times_healed = sum(map(lambda indiv: indiv.times_healed, data.generation)) / len(data.generation)
        self.last_k_avg_fitness.append(avg_times_healed)
        moving_avg_times_healed = sum(self.last_k_avg_fitness) / len(self.last_k_avg_fitness)

        if moving_avg_times_healed > self.theoretical_max_fitness * .8:
            # We've hit 80% of the theoretical max fitness, so we can stop now.
            save_project_data(self.to_data())

            for idx, indiv in enumerate(data.generation):
                save_individual_data(self.id, idx, indiv.to_data())

            is_training_complete = True

        if self.generation_num % 10 == 0:
            save_project_data(self.to_data())

            for idx, indiv in enumerate(data.generation):
                save_individual_data(self.id, idx, indiv.to_data())

        draw_data = ProjectDrawingData(
            last_sim_duration=data.last_sim_duration,
            last_training_duration=data.last_training_duration,
            avg_times_healed=avg_times_healed,
            generation_num=self.generation_num
        )

        return (draw_data, is_training_complete)

    @staticmethod
    def new(strategy_key: StrategyKey) -> "Project":
        project = Project()
        project.last_k_avg_fitness = deque(maxlen=20)
        project.theoretical_max_fitness = calculate_theoretical_max_fitness()
        project.strategy_key = strategy_key
        project.id = uuid.uuid4()
        return project

