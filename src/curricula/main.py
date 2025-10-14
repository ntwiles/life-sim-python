from src.curricula.evolutionary import apply_evolutionary_curriculum
from src.curricula.reinforcement import apply_reinforcement_curriculum
from src.curricula.types import CurriculumFn

curriculum_functions: dict[str, CurriculumFn] = {
    "evolutionary": apply_evolutionary_curriculum, 
    "reinforcement": apply_reinforcement_curriculum, 
}

