
import math
from multiprocessing import Queue
import pyglet
from pyglet import shapes, text

from shared.lib import GRID_SIZE, NUM_HEAL_ZONES, NUM_INDIVS, WINDOW_SCALE, HealZone, IndividualUpdateContext, PipeMessage

class Viewer:
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
            self.heal_zone_ids.append(shapes.Circle(0, 0, WINDOW_SCALE, color=(110, 255, 100, 60)))


        self.label = pyglet.text.Label('',
                                font_name='Times New Roman',
                                font_size=18,
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
                    # 50 denominator is arbitrary
                    percent_healed = min(update.times_healed / 50, 1)
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
        try:
            message: PipeMessage = self.queue.get()

            if message is None:
                print('Viewer done')
                pyglet.app.exit()
                return
            
            self.indiv_updates = message.indiv_updates
            self.heal_zones = message.heal_zones
        except EOFError as e:
            print(e)
            pyglet.app.exit()
            return


    def run(self):
        pyglet.app.run()


def viewer_worker(queue: Queue) -> None:
    viewer = Viewer(queue)
    viewer.run()
    print('Viewer worker done')