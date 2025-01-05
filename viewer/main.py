
import math
from multiprocessing import Queue
import pyglet
from pyglet import shapes, text
import tensorflow as tf

from shared.lib import GRID_SIZE, NUM_HEAL_ZONES, NUM_INDIVS, SIMULATOR_RUNS, SIMULATOR_STEPS, WINDOW_SCALE, HealZone, IndividualUpdateContext, PipeMessage
from simulator.main import Simulator, select_breeders, spawn_initial_generation, spawn_next_generation

class Viewer:
    sim: Simulator
    steps: int
    sims: int
    queue: Queue
    indiv_ids: list[shapes.Circle]
    heal_zone_ids: list[shapes.Circle]
    indiv_updates: list[IndividualUpdateContext] | None
    heal_zones: list[HealZone] | None
    label: text.Label
    rendering_enabled: bool
    
    def __init__(self, queue: Queue):
        self.queue = queue
        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE) # type: ignore

        self.rendering_enabled = True
                
        self.indiv_ids = []
        for _ in range(NUM_INDIVS):
            self.indiv_ids.append(shapes.Circle(0,0, WINDOW_SCALE))

        self.heal_zone_ids = []
        for _ in range(NUM_HEAL_ZONES):
            self.heal_zone_ids.append(shapes.Circle(0, 0, WINDOW_SCALE, color=(110, 255, 100, 60), segments=32))


        self.label = pyglet.text.Label('',
                                font_name='Times New Roman',
                                font_size=12,
                                x=20, y=20,
                                color=(255, 255, 255, 255),
                                anchor_x='left', anchor_y='bottom')

        self.indiv_updates = None
        self.heal_zones = None

        pyglet.clock.schedule_interval(self.update, 1/60.0)

        self.window.push_handlers(self)


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self) -> None:
        self.window.clear()

        if self.rendering_enabled:
            if self.heal_zones is not None:
                for heal_zone in self.heal_zones:
                    circle = self.heal_zone_ids[0]
                    circle.radius = heal_zone.radius * WINDOW_SCALE
                    circle.position = heal_zone.position[0] * WINDOW_SCALE, heal_zone.position[1] * WINDOW_SCALE
                    circle.draw()

            if self.indiv_updates is not None:
                times_healed = 0

                for i, update in enumerate(self.indiv_updates):
                    times_healed += update.times_healed
                    # 280 denominator is based on theoretical average maximum times healed.
                    percent_healed = min(update.times_healed / 280, 1)
                    r = 255 - math.floor(percent_healed * 255)
                    g = 255
                    b = 255

                    circle = self.indiv_ids[i]
                    circle.position = update.next_position[0] * WINDOW_SCALE, update.next_position[1] * WINDOW_SCALE
                    circle.color = (r, g, b, 255)
                    circle.draw()

                # Estimation not aligned with the simulator.
                self.label.text = f'Avg. times healed: {times_healed / NUM_INDIVS}'
                self.label.draw()
    

    def update(self, _dt: float):
        with tf.device('/GPU:0'):
            self.indiv_updates = self.sim.update(self.steps / SIMULATOR_STEPS)
            self.heal_zones = self.sim.heal_zones
            self.steps -= 1

            if self.steps == 0:
                average_times_healed = sum(map(lambda indiv: indiv.times_healed, self.sim.indivs)) / len(self.sim.indivs)
                print(f"Generation {SIMULATOR_RUNS - self.sims + 1} done. Average times healed: {average_times_healed}")

                for i, indiv in enumerate(self.sim.indivs):
                    indiv.model.save_weights(f".models/{i}.h5")

                breeders = select_breeders(self.sim.indivs)
                next_generation = spawn_next_generation(breeders)
                
                self.sim = Simulator(next_generation)
                self.sims -= 1
                self.steps = SIMULATOR_STEPS

            if self.sims == 0:
                print("Simulator done")
                pyglet.app.exit()
                return


    def run(self):
        self.sim = Simulator(spawn_initial_generation())
        self.steps = SIMULATOR_STEPS
        self.sims = SIMULATOR_RUNS
        pyglet.app.run()


def viewer_worker(queue: Queue) -> None:
    viewer = Viewer(queue)
    viewer.run()
    print('Viewer worker done')