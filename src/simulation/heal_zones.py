from config import HEAL_ZONE_RADIUS, HEAL_ZONE_COUNT
from simulation.spawning import get_closest_zone_by_position, random_circle_position

class HealZone:
    position: tuple[int, int]
    radius: int

    def __init__(self, position: tuple[int, int], radius: int):
        self.position = position
        self.radius = radius


def spawn_heal_zones() -> list[HealZone]:
    heal_zones: list[HealZone] = []

    for _ in range(HEAL_ZONE_COUNT):

        position = random_circle_position(HEAL_ZONE_RADIUS)

        if len(heal_zones) > 0:
            while get_closest_zone_by_position(heal_zones, position)[1] < HEAL_ZONE_RADIUS * 2:
                position = random_circle_position(HEAL_ZONE_RADIUS)

        heal_zones.append(HealZone(position, HEAL_ZONE_RADIUS))

    return heal_zones