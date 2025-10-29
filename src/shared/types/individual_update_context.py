from dataclasses import dataclass


@dataclass
class IndividualUpdateContext:
    heal_zone_dir: tuple[float, float]
    heal_zone_dist: float
    rad_zone_dir: tuple[float, float]
    rad_zone_dist: float
    rad_zone_move_dir: tuple[float, float]
    next_position: tuple[int, int]
    times_healed: int