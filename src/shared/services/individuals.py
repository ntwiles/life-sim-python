from dataclasses import dataclass
from uuid import UUID

import numpy as np

@dataclass
class IndividualData:
    weights: list[np.ndarray]

def save_individuals(project_id: UUID, indivs: list[IndividualData]):
    for i, indiv in enumerate(indivs):
        save_individual_data(project_id, i, indiv)

def save_individual_data(project_id: UUID, indiv_id: int, individual: IndividualData) -> None:
    np.savez_compressed(f".projects/{str(project_id)}/models/{indiv_id}.npz", *individual.weights)

def load_individual_data(project_id: UUID, indiv_id: int) -> IndividualData:
    data = np.load(f".projects/{str(project_id)}/models/{indiv_id}.npz")
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    return IndividualData(weights=weights)