from dataclasses import dataclass
from collections import deque
import json
from uuid import UUID
from filelock import FileLock

from shared.config import PLOT_MAX_DATA_POINTS
from shared.types.strategies import StrategyKey

@dataclass
class ProjectData:
    id: UUID
    last_k_avg_fitness: deque[float]
    strategy: StrategyKey
    generation_num: int

def _make_json_path(project_id: UUID) -> str:
    return f".data/projects/{str(project_id)}/project.json"

def _make_fitness_path(project_id: UUID) -> str:
    return f".data/projects/{str(project_id)}/avg_fitness.txt"

def save_project_data(project: ProjectData):
    json_path = _make_json_path(project.id)
    lock_path = json_path + ".lock"

    with FileLock(lock_path):
        with open(json_path, 'w') as file:
            data = {
                'id': str(project.id),
                'strategy': project.strategy,
                'generation_num': project.generation_num
            }
            json.dump(data, file)

        with open(_make_fitness_path(project.id), 'w') as file:
            file.write('\n'.join(str(v) for v in project.last_k_avg_fitness))

def load_project_datas() -> list[ProjectData]:
    import os
    project_dirs = [name for name in os.listdir('.data/projects') if os.path.isdir(os.path.join('.data/projects', name))]

    projects = []
    for project_dir in project_dirs:
        project = load_project_data(UUID(project_dir))
        projects.append(project)

    return projects

_PROJECTS_JSON_PATH = ".data/projects.json"

def save_selected_project_id(project_id: UUID):
    data = {}
    try:
        with open(_PROJECTS_JSON_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        pass
    data['selected_project_id'] = str(project_id)
    with open(_PROJECTS_JSON_PATH, 'w') as f:
        json.dump(data, f)

def load_selected_project_id() -> UUID | None:
    try:
        with open(_PROJECTS_JSON_PATH, 'r') as f:
            data = json.load(f)
        return UUID(data['selected_project_id'])
    except (FileNotFoundError, KeyError):
        return None

def load_project_data(project_id: UUID) -> ProjectData:
    json_path = _make_json_path(project_id)
    lock_path = json_path + ".lock"

    with FileLock(lock_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
            id = UUID(data['id'])
            strategy = data['strategy']
            generation_num = data['generation_num']

        with open(_make_fitness_path(project_id), 'r') as file:
            last_k_avg_fitness = deque(
                (float(line) for line in file if line.strip()),
                maxlen=PLOT_MAX_DATA_POINTS
            )

    return ProjectData(id=id, last_k_avg_fitness=last_k_avg_fitness, strategy=strategy, generation_num=generation_num)

