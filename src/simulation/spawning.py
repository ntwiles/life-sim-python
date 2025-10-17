
import math
from random import randint
from typing import TypeVar

from core.config import GRID_SIZE
from simulation.models import HasPosition

def random_circle_position(radius: int) -> tuple[int, int]:
    return (randint(radius, GRID_SIZE - radius), randint(radius, GRID_SIZE - radius))

T = TypeVar('T', bound=HasPosition)

def get_closest_zone_by_position(zones: list[T], position: tuple[int, int]) -> tuple[T, float]:
    closest_zone = zones[0]
    closest_zone_dist = math.dist(closest_zone.position, position)

    for zone in zones[1:]:
        dist = math.dist(zone.position, position)
        if dist < closest_zone_dist:
            closest_zone = zone
            closest_zone_dist = dist

    return (closest_zone, closest_zone_dist)

# TODO: These functions aren't use only for spawning. Come up with a better name.