from multiprocessing.connection import PipeConnection

import pyglet
from pyglet import shapes

from shared.lib import GRID_SIZE, NUM_FOOD, NUM_INDIVS, WINDOW_SCALE, Individual, PipeMessage

class Viewer:
    def __init__(self, pipe: PipeConnection):
        self.pipe = pipe
        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE)

        self.indiv_ids: list[shapes.Circle | None] = [None] * NUM_INDIVS
        self.indivs = [Individual(0, (0, 0)) for _ in range(NUM_INDIVS)]

        self.food_ids: list[shapes.Rectangle | None] = [None] * NUM_FOOD

        self.rendering_enabled = True
                
        for i in range(NUM_INDIVS):
            self.indiv_ids[i] = shapes.Circle(0,0, WINDOW_SCALE)

        for i in range(NUM_FOOD):
            self.food_ids[i] = shapes.Rectangle(0, 0, WINDOW_SCALE, WINDOW_SCALE, color=(0, 255, 0))

        pyglet.clock.schedule_interval(self.update, 1/60.0)

        self.window.push_handlers(self)


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self):
        self.window.clear()

        if self.rendering_enabled:
            for i, indiv in enumerate(self.indivs):
                circle = self.indiv_ids[i]
                circle.position = indiv.position[0] * WINDOW_SCALE, indiv.position[1] * WINDOW_SCALE
                circle.draw()

            for food in self.foods:
                rect = self.food_ids[0]
                rect.position = food[0] * WINDOW_SCALE, food[1] * WINDOW_SCALE
                rect.draw()


    def update(self, _dt: float):
        try:
            message: PipeMessage = self.pipe.recv()
            self.indivs = message.indivs
            self.foods = message.food
        except EOFError:
            print('Viewer done')
            pyglet.app.exit()


    def run(self):
        pyglet.app.run()


def viewer_worker(pipe: PipeConnection):
    viewer = Viewer(pipe)
    viewer.run()