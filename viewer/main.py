from multiprocessing.connection import PipeConnection

import pyglet
from pyglet import shapes

from shared.lib import GRID_SIZE, NUM_FOOD, NUM_INDIVS, WINDOW_SCALE, Individual, IndividualUpdateContext, PipeMessage

class Viewer:
    pipe: PipeConnection
    indiv_ids: list[shapes.Circle]
    food_ids: list[shapes.Rectangle]
    indiv_updates: list[IndividualUpdateContext]
    foods: list[tuple[int, int]]
    rendering_enabled: bool
    
    def __init__(self, pipe: PipeConnection):
        self.pipe = pipe
        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE) # type: ignore

        self.rendering_enabled = True
                
        self.indiv_ids = []
        for _ in range(NUM_INDIVS):
            self.indiv_ids.append(shapes.Circle(0,0, WINDOW_SCALE))

        self.food_ids = []
        for _ in range(NUM_FOOD):
            self.food_ids.append(shapes.Rectangle(0, 0, WINDOW_SCALE, WINDOW_SCALE, color=(0, 255, 0)))

        pyglet.clock.schedule_interval(self.update, 1/60.0)

        self.window.push_handlers(self)


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self) -> None:
        self.window.clear()

        if self.rendering_enabled:
            for i, update in enumerate(self.indiv_updates):
                circle = self.indiv_ids[i]
                circle.position = update.next_position[0] * WINDOW_SCALE, update.next_position[1] * WINDOW_SCALE
                circle.draw()

            for food in self.foods:
                rect = self.food_ids[0]
                rect.position = food[0] * WINDOW_SCALE, food[1] * WINDOW_SCALE
                rect.draw()


    def update(self, _dt: float):
        try:
            message: PipeMessage = self.pipe.recv()
            self.indiv_updates = message.indiv_updates
            self.foods = message.food
        except EOFError:
            print('Viewer done')
            pyglet.app.exit()


    def run(self):
        pyglet.app.run()


def viewer_worker(pipe: PipeConnection):
    viewer = Viewer(pipe)
    viewer.run()