

import json
from uuid import UUID

from core.config import NUM_INDIVS
from simulation.individual import Individual

def save_individuals(project_id: UUID, indivs: list[Individual]):
    for i, indiv in enumerate(indivs):
        indiv.model.inner.save_weights(f".projects/{str(project_id)}/models/{i}.weights.h5")
        with open(f".projects/{str(project_id)}/models/{i}.json", 'w') as file:
            data = {'num_generations': indiv.model.num_generations}
            json.dump(data, file)

def load_individuals(project_id: UUID) -> list[Individual]:
    return [load_individual(project_id, i) for i in range(NUM_INDIVS)]


def load_individual(project_id: UUID, indiv_id: int) -> Individual:
    indiv = Individual()

    try:
        indiv.model.inner.load_weights(f".projects/{str(project_id)}/models/{indiv_id}.weights.h5")

        with open(f".projects/{str(project_id)}/models/{indiv_id}.json", 'r') as file:
            data = json.load(file)
            indiv.model.num_generations = data['num_generations']

    except Exception as e:
        indiv.model.num_generations = 0
        print(f"Failed to load model, initializing randomly. Error: {e}")

    return indiv