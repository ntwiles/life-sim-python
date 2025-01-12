
from collections import deque
import math
import time
import pyglet
from pyglet import shapes, text
import tensorflow as tf

from config import GRID_SIZE, NUM_HEAL_ZONES, NUM_INDIVS, SIMULATOR_STEPS, WINDOW_SCALE
from src.main import IndividualUpdateContext, Simulation, select_breeders, spawn_initial_generation, spawn_next_generation

class Application:
    sim: Simulation
    steps_remaining: int

    indiv_updates: list[IndividualUpdateContext] | None
    avg_times_healed: float
    last_k_avg_times_healed: deque[float]
    moving_avg_times_healed: float
    time_started: float

    indiv_ids: list[shapes.Circle]
    heal_zone_ids: list[shapes.Circle]
    layout: text.layout.TextLayout
    stats_document: text.document.UnformattedDocument
    label: text.Label

    rendering_enabled: bool
    
    def __init__(self):
        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE) # type: ignore

        self.rendering_enabled = True
                
        self.indiv_ids = []
        for _ in range(NUM_INDIVS):
            self.indiv_ids.append(shapes.Circle(0,0, WINDOW_SCALE))

        self.heal_zone_ids = []
        for _ in range(NUM_HEAL_ZONES):
            self.heal_zone_ids.append(shapes.Circle(0, 0, WINDOW_SCALE, color=(110, 255, 100, 60), segments=32))


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

        self.indiv_updates = None
        self.avg_times_healed = 0
        self.moving_avg_times_healed = 0
        self.last_k_avg_times_healed = deque(maxlen=20)
        self.last_run_time = 0.0
        self.time_started = time.time()

        pyglet.clock.schedule_interval(self.update, 1/60.0)
        self.window.push_handlers(self)


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self) -> None:
        self.window.clear()

        if self.rendering_enabled:
            if self.sim.heal_zones is not None:
                for heal_zone in self.sim.heal_zones:
                    circle = self.heal_zone_ids[0]
                    circle.radius = heal_zone.radius * WINDOW_SCALE
                    circle.position = heal_zone.position[0] * WINDOW_SCALE, heal_zone.position[1] * WINDOW_SCALE
                    circle.draw()

            if self.indiv_updates is not None:
                for i, update in enumerate(self.indiv_updates):
                    # 280 denominator is based on theoretical average maximum times healed.
                    percent_healed = min(update.times_healed / 280, 1)
                    r = 255 - math.floor(percent_healed * 255)
                    g = 255
                    b = 255

                    circle = self.indiv_ids[i]
                    circle.position = update.next_position[0] * WINDOW_SCALE, update.next_position[1] * WINDOW_SCALE
                    circle.color = (r, g, b, 255)
                    circle.draw()

                analytics = [
                    f"Avg. fitness: { round(self.avg_times_healed, 2) }",
                    f"Moving avg. fitness: { round(self.moving_avg_times_healed, 2) }",
                    f"Steps remaining: {self.steps_remaining}",
                    f"Last run time: { round(self.last_run_time, 2) }s"
                ]

                self.stats_document.text = '\n'.join(analytics)
                self.layout.draw()
    

    def update(self, _dt: float):
        with tf.device('/GPU:0'):
            self.indiv_updates = self.sim.update(self.steps_remaining / SIMULATOR_STEPS)
            self.steps_remaining -= 1
            self.avg_times_healed = sum(map(lambda indiv: indiv.times_healed, self.sim.indivs)) / len(self.sim.indivs)

            if self.steps_remaining == 0:
                for i, indiv in enumerate(self.sim.indivs):
                    indiv.model.inner.save_weights(f".models/{i}.h5")

                breeders = select_breeders(self.sim.indivs)
                next_generation = spawn_next_generation(breeders)
                
                self.sim = Simulation(next_generation)


                self.steps_remaining = SIMULATOR_STEPS

                # TODO: Maybe this should be done in the Simulation class.
                self.last_k_avg_times_healed.append(self.avg_times_healed)
                self.moving_avg_times_healed = sum(self.last_k_avg_times_healed) / len(self.last_k_avg_times_healed)
                self.last_run_time = time.time() - self.time_started
                self.time_started = time.time()


    def run(self):
        self.sim = Simulation(spawn_initial_generation())
        self.steps_remaining = SIMULATOR_STEPS

        pyglet.app.run()