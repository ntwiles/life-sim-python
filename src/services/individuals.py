

import json
from config import NUM_INDIVS
from src.simulation.individual import Individual

def save_individuals(indivs: list[Individual]):
    for i, indiv in enumerate(indivs):
        indiv.model.inner.save_weights(f".models/{i}.weights.h5")

        with open(f".models/{i}.json", 'w') as file:
            data = {
                'num_simulations': indiv.model.num_simulations
            }

            json.dump(data, file)

def load_individuals() -> list[Individual]:
    return [load_individual(i) for i in range(NUM_INDIVS)]


def load_individual(id: int) -> Individual:
    indiv = Individual()

    try:
        indiv.model.inner.load_weights(f".models/{id}.weights.h5")

        with open(f".models/{id}.json", 'r') as file:
            data = json.load(file)
            indiv.model.num_simulations = data['num_simulations']

    except Exception as e:
        indiv.model.num_simulations = 0
        print(f"Failed to load model, initializing randomly. Error: {e}")

    return indiv