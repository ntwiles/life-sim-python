

from dataclasses import dataclass
from collections import deque
import json
from uuid import UUID

from shared.config import PLOT_MAX_DATA_POINTS
from shared.strategies.models import StrategyKey

@dataclass
class ProjectData:
    id: UUID
    last_k_avg_fitness: deque[float]
    strategy: StrategyKey
    generation_num: int

def save_project_data(project: ProjectData):
    with open(f".projects/{str(project.id)}/project.json", 'w') as file:
        data = {
            'last_k_avg_fitness': list(project.last_k_avg_fitness),
            'id': str(project.id),
            'strategy': project.strategy,
            'generation_num': project.generation_num
        }

        json.dump(data, file)

def load_project_datas() -> list[ProjectData]:
    import os
    project_dirs = [name for name in os.listdir('.projects') if os.path.isdir(os.path.join('.projects', name))]

    projects = []
    for project_dir in project_dirs:
        project = load_project_data(UUID(project_dir))
        projects.append(project)

    return projects

def load_project_data(project_id: UUID) -> ProjectData:
    with open(f".projects/{str(project_id)}/project.json", 'r') as file:
        data = json.load(file)
        last_k_avg_fitness = deque(data['last_k_avg_fitness'], maxlen=PLOT_MAX_DATA_POINTS)
        id = UUID(data['id'])
        strategy = data['strategy']
        generation_num = data['generation_num']

    return ProjectData(id=id, last_k_avg_fitness=last_k_avg_fitness, strategy=strategy, generation_num=generation_num)

