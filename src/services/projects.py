

from dataclasses import dataclass
from collections import deque
import json
from uuid import UUID

from src.curricula.types import CurriculumKey

@dataclass
class ProjectData:
    id: UUID
    last_k_avg_times_healed: deque[float]
    curriculum: CurriculumKey

def save_project(project: ProjectData):
    with open(f".projects/{str(project.id)}/project.json", 'w') as file:
        data = {
            'last_k_avg_times_healed': list(project.last_k_avg_times_healed),
            'id': str(project.id),
            'curriculum': project.curriculum
        }

        json.dump(data, file)

def load_projects() -> list[ProjectData]:
    import os
    project_dirs = [name for name in os.listdir('.projects') if os.path.isdir(os.path.join('.projects', name))]

    projects = []
    for project_dir in project_dirs:
        project = load_project(UUID(project_dir))
        projects.append(project)

    return projects

def load_project(project_id: UUID) -> ProjectData:
    with open(f".projects/{str(project_id)}/project.json", 'r') as file:
        data = json.load(file)
        last_k_avg_times_healed = deque(data['last_k_avg_times_healed'], maxlen=20)
        id = UUID(data['id'])
        curriculum = data['curriculum']

    return ProjectData(id=id, last_k_avg_times_healed=last_k_avg_times_healed, curriculum=curriculum)

