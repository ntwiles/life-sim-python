
import math
from random import randint
from typing import TypeVar

from config import GRID_SIZE
from src.simulation.types import HasPosition

def random_circle_position(radius: int) -> tuple[int, int]:
    return (randint(radius, GRID_SIZE - radius), randint(radius, GRID_SIZE - radius))

T = TypeVar('T', bound=HasPosition)

def get_closest_zone_by_position(zones: list[HasPosition], position: tuple[int, int]) -> tuple[T, float]:
    closest_zone = zones[0]
    closest_zone_dist = math.dist(closest_zone.position, position)

    for zone in zones[1:]:
        dist = math.dist(zone.position, position)
        if dist < closest_zone_dist:
            closest_zone = zone
            closest_zone_dist = dist

    return (closest_zone, closest_zone_dist)