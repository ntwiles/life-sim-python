# TODO: Consider generalizing with heal_zone implementation.

from random import randint

from core.config import GRID_SIZE, RAD_ZONE_RADIUS, RAD_ZONE_MOVE_DELAY, RAD_ZONE_COUNT, RAD_ZONE_ENABLE_MOVEMENT
from simulation.spawning import get_closest_zone_by_position, random_circle_position

class RadZone:
    position: tuple[int, int]
    direction: tuple[int, int]
    radius: int
    time_since_last_move: int

    def __init__(self, position: tuple[int, int], radius: int):
        self.position = position
        self.radius = radius

        dir_x = randint(-1, 1)
        dir_y = randint(-1, 1)
        self.direction = (dir_x, dir_y)
        self.time_since_last_move = 0
    
    def update(self):
        self.time_since_last_move += 1

        if not RAD_ZONE_ENABLE_MOVEMENT:
            return

        if self.time_since_last_move < RAD_ZONE_MOVE_DELAY:
            return
        
        dest_pos = (self.position[0] + self.direction[0], self.position[1] + self.direction[1])
            
        half_radius = self.radius / 2
        if dest_pos[0] - half_radius < 0 or dest_pos[0] + half_radius >= GRID_SIZE:
            self.direction = (-self.direction[0], self.direction[1])
            dest_pos = (self.position[0] + self.direction[0], self.position[1] + self.direction[1])
        
        if dest_pos[1] - half_radius < 0 or dest_pos[1] + half_radius >= GRID_SIZE:
            self.direction = (self.direction[0], -self.direction[1])
            dest_pos = (self.position[0] + self.direction[0], self.position[1] + self.direction[1])

        self.position = dest_pos
        self.time_since_last_move = 0

def spawn_rad_zones() -> list[RadZone]:
    rad_zones: list[RadZone] = []

    for _ in range(RAD_ZONE_COUNT):

        position = random_circle_position(RAD_ZONE_RADIUS)

        if len(rad_zones) > 0:
            while get_closest_zone_by_position(rad_zones, position)[1] < RAD_ZONE_RADIUS * 2:
                position = random_circle_position(RAD_ZONE_RADIUS)

        rad_zones.append(RadZone(position, RAD_ZONE_RADIUS))

    return rad_zones