from dataclasses import dataclass
import math
from multiprocessing.connection import PipeConnection
from random import randint

from shared.lib import GRID_SIZE, HEAL_ZONE_RADIUS, NUM_HEAL_ZONES, NUM_INDIVS, SIMULATOR_STEPS, HealZone, Individual, IndividualUpdateContext, PipeMessage
from simulator.network.main import decide

class Simulator:
    generation_time: int
    indivs: list[Individual]
    heal_zones: list[HealZone]

    def __init__(self):
        self.generation_time = 0

        self.indivs = spawn_indivs()
        self.heal_zones = spawn_heal_zones()

    def update(self, generation_time: int) -> list[IndividualUpdateContext]:
        return list(map(lambda indiv: self.update_individual(indiv, generation_time), self.indivs))
    

    def get_closest_heal_zone(self, position: tuple[int, int]) -> HealZone:
        closest_heal_zone = None
        closest_heal_zone_dist = float('inf')

        for heal_zone in self.heal_zones:
            dist = math.dist(heal_zone.position, position)
            if dist < closest_heal_zone_dist:
                closest_heal_zone = heal_zone
                closest_heal_zone_dist = dist

        return closest_heal_zone
    
    def update_individual(self, indiv: Individual, generation_time: int) -> IndividualUpdateContext:
        heal_zone = self.get_closest_heal_zone(indiv.position)

        heal_zone_disp = (heal_zone.position[0] - indiv.position[0], heal_zone.position[1] - indiv.position[1])
        heal_zone_angle = math.atan2(heal_zone_disp[1], heal_zone_disp[0])

        context = IndividualUpdateContext(heal_zone_angle, indiv.position)

        decision = decide(indiv, context, generation_time)
        next_position = (indiv.position[0] + decision[0], indiv.position[1] + decision[1])

        context.next_position = next_position
        indiv.position = next_position

        return context

def spawn_heal_zones() -> list[HealZone]:
    heal_zones = []
    for _ in range(NUM_HEAL_ZONES):
        position = (randint(0, GRID_SIZE), randint(0, GRID_SIZE))
        heal_zones.append(HealZone(position, HEAL_ZONE_RADIUS))

    return heal_zones

def spawn_indivs() -> list[Individual]:
    return list(map(lambda id: Individual(id, (randint(0, GRID_SIZE), randint(0, GRID_SIZE))), range(NUM_INDIVS)))

def simulator_worker(pipe: PipeConnection) -> None:
    sim = Simulator()
    steps = SIMULATOR_STEPS

    while steps > 0:
        indiv_updates = sim.update(SIMULATOR_STEPS - steps)

        pipe.send(PipeMessage(indiv_updates, sim.heal_zones))
        steps -= 1

    pipe.close()
    print("Simulator done")


