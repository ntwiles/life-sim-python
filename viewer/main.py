from multiprocessing.connection import PipeConnection

import pyglet
from pyglet import shapes

from shared.lib import GRID_SIZE, NUM_INDIVS, WINDOW_SCALE, Individual

class Viewer:
    def __init__(self, pipe: PipeConnection):
        self.pipe = pipe
        self.window = pyglet.window.Window()

        self.ids: list[shapes.Circle | None] = [None] * NUM_INDIVS
        self.indivs = [Individual(0, (0, 0)) for _ in range(NUM_INDIVS)]

        self.rendering_enabled = True

        self.rendering_enabled = True
                
        for i, indiv in enumerate(self.indivs):
            self.ids[i] = shapes.Circle(*indiv.position, WINDOW_SCALE)

        pyglet.clock.schedule_interval(self.update, 1/60.0)

        self.window.push_handlers(self)


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self):
        self.window.clear()

        if self.rendering_enabled:
            for i, indiv in enumerate(self.indivs):
                circle = self.ids[i]
                circle.position = indiv.position
                circle.draw()


    def update(self, _dt):
        try:
            self.indivs = self.pipe.recv()
        except EOFError:
            print('Viewer done')
            pyglet.app.exit()


    def run(self):
        pyglet.app.run()


def viewer_worker(pipe: PipeConnection):
    viewer = Viewer(pipe)
    viewer.run()