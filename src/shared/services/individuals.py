from dataclasses import dataclass
from uuid import UUID
from filelock import FileLock
import os

import numpy as np

@dataclass
class IndividualData:
    weights: list[np.ndarray]

def _ensure_models_dir(project_id: UUID):
    models_dir = f".data/projects/{str(project_id)}/models"
    os.makedirs(models_dir, exist_ok=True)

def _make_file_path(project_id: UUID, indiv_id: int) -> str:
    return f".data/projects/{str(project_id)}/models/{indiv_id}.npz"

def save_individuals(project_id: UUID, indivs: list[IndividualData]):
    _ensure_models_dir(project_id)
    for i, indiv in enumerate(indivs):
        save_individual_data(project_id, i, indiv)

def save_individual_data(project_id: UUID, indiv_id: int, individual: IndividualData) -> None:
    _ensure_models_dir(project_id)

    file_path = _make_file_path(project_id, indiv_id)
    lock_path = file_path + ".lock"

    with FileLock(lock_path):
        np.savez_compressed(file_path, *individual.weights)

def load_individual_data(project_id: UUID, indiv_id: int) -> IndividualData:
    file_path = _make_file_path(project_id, indiv_id)
    lock_path = file_path + ".lock"

    with FileLock(lock_path):
        data = np.load(file_path)
        weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        return IndividualData(weights=weights)