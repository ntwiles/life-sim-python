from multiprocessing.connection import PipeConnection
from random import randint

from shared.lib import NUM_INDIVS, Individual

def spawn_indivs():
    return list(map(lambda id: Individual(id, (randint(0, 1000), randint(0, 1000))), range(NUM_INDIVS)))

def update_indivs(indivs: list[Individual]):
    for indiv in indivs:
        indiv.position = (indiv.position[0] + 1, indiv.position[1] + 1)

def simulator_worker(pipe: PipeConnection):
    indivs = spawn_indivs()

    steps = 1000

    while steps > 0:
        update_indivs(indivs)

        pipe.send(indivs)
        steps -= 1

    pipe.close()
    print("Simulator done")


