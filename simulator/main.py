from dataclasses import dataclass
import math
from multiprocessing.connection import PipeConnection
from random import randint

from scipy.spatial import cKDTree

from shared.lib import GRID_SIZE, NUM_FOOD, NUM_INDIVS, Individual, IndividualUpdateContext, PipeMessage


def spawn_food():
    food = []
    for _ in range(NUM_FOOD):
        food.append((randint(0, GRID_SIZE), randint(0, GRID_SIZE)))

    return food

def spawn_indivs():
    return list(map(lambda id: Individual(id, (randint(0, GRID_SIZE), randint(0, GRID_SIZE))), range(NUM_INDIVS)))

def update_individual(indiv: Individual, food_kd_tree: cKDTree, foods: list[tuple[int, int]]):
    _dist, idx = food_kd_tree.query(indiv.position)
    food = foods[idx]
    food_disp = (food[0] - indiv.position[0], food[1] - indiv.position[1])
    food_angle = math.atan2(food_disp[1], food_disp[0])

    next_position = (indiv.position[0] + 1, indiv.position[1] + 1)

    indiv.position = next_position

    return IndividualUpdateContext(food_angle, next_position)

def update(indivs: list[Individual], foods: list[tuple[int, int]], food_kd_tree: cKDTree):
    return list(map(lambda indiv: update_individual(indiv, food_kd_tree, foods), indivs))

def simulator_worker(pipe: PipeConnection):
    indivs = spawn_indivs()
    foods = spawn_food()
    food_kd_tree = cKDTree(foods)

    steps = 1000

    while steps > 0:
        indiv_updates = update(indivs, foods, food_kd_tree)

        pipe.send(PipeMessage(indiv_updates, foods))
        steps -= 1

    pipe.close()
    print("Simulator done")


