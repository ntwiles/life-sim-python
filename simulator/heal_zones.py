import math
from random import randint
from shared.lib import GRID_SIZE, HEAL_ZONE_RADIUS, NUM_HEAL_ZONES, HealZone


def get_closest_heal_zone(heal_zones: list[HealZone], position: tuple[int, int]) -> tuple[HealZone, float]:
    closest_heal_zone = heal_zones[0]
    closest_heal_zone_dist = float('inf')

    for heal_zone in heal_zones:
        dist = math.dist(heal_zone.position, position)
        if dist < closest_heal_zone_dist:
            closest_heal_zone = heal_zone
            closest_heal_zone_dist = dist

    return (closest_heal_zone, closest_heal_zone_dist)

def random_heal_zone_position() -> tuple[int, int]:
    return (randint(HEAL_ZONE_RADIUS, GRID_SIZE - HEAL_ZONE_RADIUS), randint(HEAL_ZONE_RADIUS, GRID_SIZE - HEAL_ZONE_RADIUS))

def spawn_heal_zones() -> list[HealZone]:
    heal_zones: list[HealZone] = []

    for _ in range(NUM_HEAL_ZONES):

        position = random_heal_zone_position()

        if len(heal_zones) > 0:
            while get_closest_heal_zone(heal_zones, position)[1] < HEAL_ZONE_RADIUS * 2:
                position = random_heal_zone_position()

        heal_zones.append(HealZone(position, HEAL_ZONE_RADIUS))

    return heal_zones