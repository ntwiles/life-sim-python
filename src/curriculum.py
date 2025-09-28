from collections import deque
import math
import time

import tensorflow as tf
from pyglet import text, shapes


from config import GRID_SIZE, NUM_HEAL_ZONES, NUM_INDIVS, NUM_RAD_ZONES, SIMULATOR_STEPS, WINDOW_SCALE
from src.fitness import calculate_theoretical_max_fitness
from src.services.individuals import save_individuals
from src.simulation.individual import IndividualUpdateContext
from src.simulation.main import Simulation, select_breeders, spawn_initial_generation, spawn_next_generation

class Curriculum:
    sim: Simulation
    steps_remaining: int
    indiv_updates: list[IndividualUpdateContext] | None

    avg_times_healed: float
    last_k_avg_times_healed: deque[float]
    moving_avg_times_healed: float
    theoretical_max_fitness: float

    time_started: float
    last_run_time: float

    stats_document: text.document.UnformattedDocument

    layout: text.layout.TextLayout
    heal_zone_ids: list[shapes.Circle]
    rad_zone_ids: list[shapes.Circle]
    indiv_ids: list[shapes.Circle]

    def __init__(self):
        self.sim = Simulation(spawn_initial_generation())
        self.steps_remaining = SIMULATOR_STEPS
        self.indiv_updates = None

        self.avg_times_healed = 0
        self.moving_avg_times_healed = 0
        self.last_k_avg_times_healed = deque(maxlen=20)

        self.theoretical_max_fitness = calculate_theoretical_max_fitness()

        self.indiv_ids = []
        for _ in range(NUM_INDIVS):
            self.indiv_ids.append(shapes.Circle(0,0, WINDOW_SCALE))

        self.heal_zone_ids = []
        for _ in range(NUM_HEAL_ZONES):
            self.heal_zone_ids.append(shapes.Circle(0, 0, WINDOW_SCALE, color=(110, 255, 100, 60), segments=32))

        self.rad_zone_ids = []
        for _ in range(NUM_RAD_ZONES):
            self.rad_zone_ids.append(shapes.Circle(0, 0, WINDOW_SCALE, color=(255, 100, 100, 60), segments=32))
            
        style = dict(
            margin_left="10px",
            margin_top="10px",
            font_name='Times New Roman',
            font_size=12,
            color=(255, 255, 255, 255)
        )

        self.stats_document = text.decode_text('')
        self.stats_document.set_style(0, len(self.stats_document.text), style)
        self.layout = text.layout.TextLayout(self.stats_document, GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE, multiline=True)

        self.last_run_time = 0.0
        self.time_started = time.time()

    # TODO: Get a better understanding of this `with` usage here. Do we actually need the GPU for all this logic?
    def update(self) -> bool:
        with tf.device('/GPU:0'):
            self.indiv_updates = self.sim.update(self.steps_remaining / SIMULATOR_STEPS)
            self.steps_remaining -= 1
            self.avg_times_healed = sum(map(lambda indiv: indiv.times_healed, self.sim.indivs)) / len(self.sim.indivs)

            # TODO: Optimize this to speed up the simulation loop. Saving models can be done async, but this isn't the biggest bottleneck.
            if self.steps_remaining == 0:
                for indiv in self.sim.indivs:
                    indiv.model.num_simulations += 1

                save_individuals(self.sim.indivs)

                breeders = select_breeders(self.sim.indivs)
                next_generation = spawn_next_generation(breeders)
                
                self.sim = Simulation(next_generation)

                self.steps_remaining = SIMULATOR_STEPS

                # TODO: Maybe this should be done in the Simulation class.
                self.last_k_avg_times_healed.append(self.avg_times_healed)
                self.moving_avg_times_healed = sum(self.last_k_avg_times_healed) / len(self.last_k_avg_times_healed)
                self.last_run_time = time.time() - self.time_started
                self.time_started = time.time()

            # We've hit 80% of the theoretical max fitness, so we can stop now.
            if self.moving_avg_times_healed > self.theoretical_max_fitness * .8:
                return False
            return True
        
    def draw(self):
        if self.sim.heal_zones is not None:
            for heal_zone in self.sim.heal_zones:
                    # TODO: Why are we just grabbing the first circle instance? If this works, maybe we should only 
                    # make a single instance altogether and use it as a brush. Same goes for rad_zones.
                    circle = self.heal_zone_ids[0]
                    circle.radius = heal_zone.radius * WINDOW_SCALE
                    circle.position = heal_zone.position[0] * WINDOW_SCALE, heal_zone.position[1] * WINDOW_SCALE
                    circle.draw()

        if self.sim.rad_zones is not None:
            for rad_zone in self.sim.rad_zones:
                circle = self.rad_zone_ids[0]
                circle.radius = rad_zone.radius * WINDOW_SCALE
                circle.position = rad_zone.position[0] * WINDOW_SCALE, rad_zone.position[1] * WINDOW_SCALE
                circle.draw()

        if self.indiv_updates is not None:
            for i, update in enumerate(self.indiv_updates):
                percent_healed = min(abs(update.times_healed) / self.theoretical_max_fitness, 1)

                r = 0
                g = 0
                b = 0
                
                if (update.times_healed > 0): 
                    r = 255 - math.floor(percent_healed * 255)
                    g = 255
                    b = 255 - math.floor(percent_healed * 255)
                else: 
                    r = 255
                    g = 255 - math.floor(percent_healed * 255)
                    b = 255 - math.floor(percent_healed * 255)

                circle = self.indiv_ids[i]
                circle.position = update.next_position[0] * WINDOW_SCALE, update.next_position[1] * WINDOW_SCALE
                circle.color = (r, g, b, 255)
                circle.draw()

            analytics = [
                f"Avg. fitness: { round(self.avg_times_healed, 2) }",
                f"Moving avg. fitness: { round(self.moving_avg_times_healed, 2) }",
                f"Steps remaining: {self.steps_remaining}",
                f"Last run time: { round(self.last_run_time, 2) }s",
                f"Num simulations: { self.sim.indivs[0].model.num_simulations }"
            ]

            self.stats_document.text = '\n'.join(analytics)

        self.layout.draw()