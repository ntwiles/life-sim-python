from typing import Callable
import numpy as np
import tensorflow as tf

from visualization.drawing_data import SimulationDrawingData
from config import GRID_SIZE, SIM_ALLOW_OOB
from model.propagation import batch_decide
from simulation.rad_zones import RadZone, spawn_rad_zones
from simulation.heal_zones import HealZone, spawn_heal_zones
from simulation.individual import Individual, IndividualUpdateContext

class Simulation:
    indivs: list[Individual]
    heal_zones: list[HealZone]
    rad_zones: list[RadZone]

    indiv_updates: list[IndividualUpdateContext] | None
    steps_remaining: int


    def __init__(self, indivs: list[Individual], on_update: Callable[[SimulationDrawingData], None] | None = None):
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


    def update(self):
        for rad_zone in self.rad_zones:
            rad_zone.update()

        contexts = []
        all_inputs = []

        for indiv in self.indivs:
            context = indiv.update(self.heal_zones, self.rad_zones)
            all_inputs.append(indiv.calculate_input_values(context))
            contexts.append(context)

        with tf.device('/GPU:0'):
            all_decisions = batch_decide([indiv.model for indiv in self.indivs], np.array(all_inputs))

        for i, (indiv, decision, context) in enumerate(zip(self.indivs, all_decisions, contexts)):
            indiv.previous_position = indiv.position
            new_position = (indiv.position[0] + decision[0], indiv.position[1] + decision[1])

            if not SIM_ALLOW_OOB:
                new_position = (
                    max(0, min(new_position[0], GRID_SIZE - 1)),
                    max(0, min(new_position[1], GRID_SIZE - 1))
                )
                
            indiv.position = new_position

            context.next_position = indiv.position

            input = all_inputs[i]
            indiv.log_step(input, decision)

        if self.on_update is not None:
            drawing_data = SimulationDrawingData(
                indiv_updates=contexts, 
                heal_zones=self.heal_zones, 
                rad_zones=self.rad_zones, 
                steps_remaining=self.steps_remaining, 
                model_num_generations=self.indivs[0].model.num_generations
            )

            self.on_update(drawing_data)

