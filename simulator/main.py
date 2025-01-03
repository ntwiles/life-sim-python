from dataclasses import dataclass
import math
from multiprocessing import Queue
from random import randint

import tensorflow as tf

from shared.lib import GRID_SIZE, HEAL_ZONE_RADIUS, INPUT_SIZE, MAX_LENGTH, NUM_HEAL_ZONES, NUM_INDIVS, SIMULATOR_RUNS, SIMULATOR_STEPS, HealZone, Individual, IndividualUpdateContext, PipeMessage
from simulator.network.main import decide

class Simulator:
    generation_time: int
    indivs: list[Individual]
    heal_zones: list[HealZone]

    def __init__(self, indivs: list[Individual]):
        self.generation_time = 0

        self.indivs = indivs
        self.heal_zones = spawn_heal_zones()

    def update(self, t: int) -> list[IndividualUpdateContext]:
        return list(map(lambda indiv: self.update_individual(indiv, t), self.indivs))
    

    def get_closest_heal_zone(self, position: tuple[int, int]) -> HealZone:
        closest_heal_zone = None
        closest_heal_zone_dist = float('inf')

        for heal_zone in self.heal_zones:
            dist = math.dist(heal_zone.position, position)
            if dist < closest_heal_zone_dist:
                closest_heal_zone = heal_zone
                closest_heal_zone_dist = dist

        return (closest_heal_zone, closest_heal_zone_dist)
    
    def update_individual(self, indiv: Individual, t: int) -> IndividualUpdateContext:
        heal_zone, heal_zone_dist = self.get_closest_heal_zone(indiv.position)

        # TODO: Calculate this in `get_closest_heal_zone`.
        heal_zone_disp = (heal_zone.position[0] - indiv.position[0], heal_zone.position[1] - indiv.position[1])
        heal_zone_angle = math.atan2(heal_zone_disp[1], heal_zone_disp[0])

        if heal_zone_dist < heal_zone.radius:
            indiv.times_healed += 1

        context = IndividualUpdateContext(heal_zone_angle, heal_zone_dist / MAX_LENGTH, indiv.position, indiv.times_healed)

        decision = decide(indiv, context, t)
        next_position = (indiv.position[0] + decision[0], indiv.position[1] + decision[1])

        context.next_position = next_position
        indiv.position = next_position

        return context

def spawn_heal_zones() -> list[HealZone]:
    heal_zones = []
    for _ in range(NUM_HEAL_ZONES):
        position = (randint(HEAL_ZONE_RADIUS, GRID_SIZE - HEAL_ZONE_RADIUS), randint(HEAL_ZONE_RADIUS, GRID_SIZE - HEAL_ZONE_RADIUS))
        heal_zones.append(HealZone(position, HEAL_ZONE_RADIUS))

    return heal_zones

def spawn_indivs() -> list[Individual]:
    return list(map(lambda _: Individual((randint(0, GRID_SIZE), randint(0, GRID_SIZE))), range(NUM_INDIVS)))

def mutate_weights(model: tf.keras.Sequential, mutation_rate=0.1):
    for var in model.trainable_variables:
        mutation_mask = tf.random.uniform(var.shape) < mutation_rate
        random_mutations = tf.random.normal(var.shape, mean=0.0, stddev=0.1)
        var.assign(tf.where(mutation_mask, var + random_mutations, var))

def simulator_worker(queue: Queue) -> None:
    sim = Simulator(spawn_indivs())
    steps = SIMULATOR_STEPS
    sims = SIMULATOR_RUNS

    with tf.device('/GPU:0'):       
        while sims > 0:
            while steps > 0:
                indiv_updates = sim.update(steps / SIMULATOR_STEPS)
                queue.put(PipeMessage(indiv_updates, sim.heal_zones))
                steps -= 1

            average_times_healed = sum(map(lambda indiv: indiv.times_healed, sim.indivs)) / len(sim.indivs)
            print(f"Generation {SIMULATOR_RUNS - sims + 1} done. Average times healed: {average_times_healed}")

            sims -= 1
            steps = SIMULATOR_STEPS


            sim.indivs.sort(key=lambda indiv: indiv.times_healed, reverse=True)

            num_breeders = math.floor(len(sim.indivs) * 0.2)
            breeders = sim.indivs[:num_breeders]

            next_generation = []
            for parent in breeders:
                for _ in range(5):
                    position = (randint(0, GRID_SIZE), randint(0, GRID_SIZE))
                    child = Individual(position)
                    child.model.set_weights(parent.model.get_weights())
                    
                    mutate_weights(child.model)
                    next_generation.append(child)
            
            sim = Simulator(next_generation)
    
    queue.put(None)
    print("Simulator done")


