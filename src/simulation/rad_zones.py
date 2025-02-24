# TODO: Consider generalizing with heal_zone implementation.

import math
from random import randint

from config import GRID_SIZE, RAD_ZONE_RADIUS, NUM_RAD_ZONES

class RadZone:
    position: tuple[int, int]
    radius: int

    def __init__(self, position: tuple[int, int], radius: int):
        self.position = position
        self.radius = radius

def get_closest_rad_zone(rad_zones: list[RadZone], position: tuple[int, int]) -> tuple[RadZone, float]:
    closest_rad_zone = rad_zones[0]
    closest_rad_zone_dist = float('inf')

    for rad_zone in rad_zones:
        dist = math.dist(rad_zone.position, position)
        if dist < closest_rad_zone_dist:
            closest_rad_zone = rad_zone
            closest_rad_zone_dist = dist

    return (closest_rad_zone, closest_rad_zone_dist)

def random_rad_zone_position() -> tuple[int, int]:
    return (randint(RAD_ZONE_RADIUS, GRID_SIZE - RAD_ZONE_RADIUS), randint(RAD_ZONE_RADIUS, GRID_SIZE - RAD_ZONE_RADIUS))

def spawn_rad_zones() -> list[RadZone]:
    rad_zones: list[RadZone] = []

    for _ in range(NUM_RAD_ZONES):

        position = random_rad_zone_position()

        if len(rad_zones) > 0:
            while get_closest_rad_zone(rad_zones, position)[1] < RAD_ZONE_RADIUS * 2:
                position = random_rad_zone_position()

        rad_zones.append(RadZone(position, RAD_ZONE_RADIUS))

    return rad_zones