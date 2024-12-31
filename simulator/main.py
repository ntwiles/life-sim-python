from dataclasses import dataclass
import math
from multiprocessing.connection import PipeConnection
from random import randint

from scipy.spatial import cKDTree

from shared.lib import GRID_SIZE, NUM_FOOD, NUM_INDIVS, SIMULATOR_STEPS, Individual, IndividualUpdateContext, PipeMessage
from simulator.network.main import decide

class Simulator:
    generation_time: int

    def __init__(self):
        self.generation_time = 0

        self.indivs = spawn_indivs()
        self.foods = spawn_food()
        self.food_kd_tree = cKDTree(self.foods)

    def update(self, generation_time: int) -> list[IndividualUpdateContext]:
        return list(map(lambda indiv: self.update_individual(indiv, generation_time), self.indivs))
    
    def update_individual(self, indiv: Individual, generation_time: int) -> IndividualUpdateContext:
        _dist, idx = self.food_kd_tree.query(indiv.position)
        food = self.foods[idx]
        food_disp = (food[0] - indiv.position[0], food[1] - indiv.position[1])
        food_angle = math.atan2(food_disp[1], food_disp[0])

        context = IndividualUpdateContext(food_angle, indiv.position)

        decision = decide(indiv, context, generation_time)

        next_position = (indiv.position[0] + decision[0], indiv.position[1] + decision[1])

        indiv.position = next_position

        return IndividualUpdateContext(food_angle, next_position)


def spawn_food() -> list[tuple[int, int]]:
    food = []
    for _ in range(NUM_FOOD):
        food.append((randint(0, GRID_SIZE), randint(0, GRID_SIZE)))

    return food

def spawn_indivs():
    return list(map(lambda id: Individual(id, (randint(0, GRID_SIZE), randint(0, GRID_SIZE))), range(NUM_INDIVS)))

def simulator_worker(pipe: PipeConnection) -> None:
    sim = Simulator()
    steps = SIMULATOR_STEPS

    while steps > 0:
        indiv_updates = sim.update(SIMULATOR_STEPS - steps)

        pipe.send(PipeMessage(indiv_updates, sim.foods))
        steps -= 1

    pipe.close()
    print("Simulator done")


