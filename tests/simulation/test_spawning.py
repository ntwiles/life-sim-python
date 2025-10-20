import math
import pytest

from simulation.heal_zones import HealZone
from simulation.spawning import get_closest_zone_by_position

@pytest.mark.parametrize("position, zones, expected_zone_index", [
    ((-10, -10), [HealZone(position=(0, 0), radius=3), HealZone(position=(-5, -5), radius=3)], 1),
    ((0, 0), [HealZone(position=(5, 5), radius=3), HealZone(position=(10, 10), radius=3)], 0),
    ((20, 20), [HealZone(position=(15, 15), radius=3), HealZone(position=(25, 25), radius=3)], 0),
    ((7, 8), [HealZone(position=(5, 5), radius=3), HealZone(position=(10, 10), radius=3)], 0),
])
def test_get_closest_zone_by_position(position, zones, expected_zone_index):
    closest_zone, closest_dist = get_closest_zone_by_position(zones, position)

    expected_zone = zones[expected_zone_index]
    expected_dist = math.dist(position, expected_zone.position)

    assert closest_zone == expected_zone
    assert closest_dist == expected_dist