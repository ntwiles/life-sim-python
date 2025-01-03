from multiprocessing.connection import PipeConnection

import pyglet
from pyglet import shapes

from shared.lib import GRID_SIZE, NUM_HEAL_ZONES, NUM_INDIVS, WINDOW_SCALE, HealZone, IndividualUpdateContext, PipeMessage

class Viewer:
    pipe: PipeConnection
    indiv_ids: list[shapes.Circle]
    heal_zone_ids: list[shapes.Circle]
    indiv_updates: list[IndividualUpdateContext] | None
    heal_zones: list[HealZone] | None
    rendering_enabled: bool
    
    def __init__(self, pipe: PipeConnection):
        self.pipe = pipe
        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE) # type: ignore

        self.rendering_enabled = True
                
        self.indiv_ids = []
        for _ in range(NUM_INDIVS):
            self.indiv_ids.append(shapes.Circle(0,0, WINDOW_SCALE))

        self.heal_zone_ids = []
        for _ in range(NUM_HEAL_ZONES):
            self.heal_zone_ids.append(shapes.Circle(0, 0, WINDOW_SCALE, color=(0, 255, 0, 60)))

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
            if self.indiv_updates is not None:
                for i, update in enumerate(self.indiv_updates):
                    circle = self.indiv_ids[i]
                    circle.position = update.next_position[0] * WINDOW_SCALE, update.next_position[1] * WINDOW_SCALE
                    circle.draw()

            if self.heal_zones is not None:
                for heal_zone in self.heal_zones:
                    circle = self.heal_zone_ids[0]
                    circle.radius = heal_zone.radius * WINDOW_SCALE
                    circle.position = heal_zone.position[0] * WINDOW_SCALE, heal_zone.position[1] * WINDOW_SCALE
                    circle.draw()


    def update(self, _dt: float):
        try:
            message: PipeMessage = self.pipe.recv()
            self.indiv_updates = message.indiv_updates
            self.heal_zones = message.heal_zones
        except EOFError:
            print('Viewer done')
            pyglet.app.exit()


    def run(self):
        pyglet.app.run()


def viewer_worker(pipe: PipeConnection):
    viewer = Viewer(pipe)
    viewer.run()