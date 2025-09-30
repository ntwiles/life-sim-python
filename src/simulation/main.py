from typing import Callable
import numpy as np
import tensorflow as tf

from src.drawing_data import DrawingData
from config import GRID_SIZE
from src.model.propagation import batch_decide
from src.simulation.rad_zones import RadZone, spawn_rad_zones
from src.simulation.heal_zones import HealZone, spawn_heal_zones
from src.simulation.individual import Individual, IndividualUpdateContext

class Simulation:
    indivs: list[Individual]
    heal_zones: list[HealZone]
    rad_zones: list[RadZone]

    indiv_updates: list[IndividualUpdateContext] | None
    steps_remaining: int


    def __init__(self, indivs: list[Individual], on_update: Callable[[DrawingData], None] | None = None):
        self.indivs = indivs
        self.heal_zones = spawn_heal_zones()
        self.rad_zones = spawn_rad_zones()
        self.indiv_updates = None
        self.steps_remaining = 0
        self.on_update = on_update
       

    def run(self, num_steps: int):
        self.steps_remaining = num_steps
        while self.steps_remaining > 0:
            self.update()
            self.steps_remaining -= 1


    # TODO: Get a better understanding of this `with` usage here. Do we actually need the GPU for all this logic?
    def update(self):
        with tf.device('/GPU:0'):
            for rad_zone in self.rad_zones:
                rad_zone.update()

            contexts = []
            all_inputs = []

            for indiv in self.indivs:
                context = indiv.update(self.heal_zones, self.rad_zones)
                all_inputs.append(indiv.calculate_input_values(context))
                contexts.append(context)

            all_decisions = batch_decide([indiv.model for indiv in self.indivs], np.array(all_inputs))

            for indiv, decision, context in zip(self.indivs, all_decisions, contexts):
                indiv.previous_position = indiv.position
                new_position = (indiv.position[0] + decision[0], indiv.position[1] + decision[1])

                indiv.position = (
                    max(0, min(new_position[0], GRID_SIZE - 1)),
                    max(0, min(new_position[1], GRID_SIZE - 1))
                )

                context.next_position = indiv.position 

            if self.on_update is not None:
                drawing_data = DrawingData(indiv_updates=contexts, heal_zones=self.heal_zones, rad_zones=self.rad_zones)
                self.on_update(drawing_data)

