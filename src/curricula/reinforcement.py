from src.curricula.types import CurriculumFn
from src.simulation.individual import Individual


def _apply_reinforcement_curriculum(generation: list[Individual]) -> list[Individual]:
    return generation

apply_reinforcement_curriculum: CurriculumFn = _apply_reinforcement_curriculum