
import pyglet
from pyglet import shapes, text

from config import GRID_SIZE, WINDOW_SCALE
from src.curriculum import Curriculum

class Application:
    curriculum: Curriculum

    indiv_ids: list[shapes.Circle]
    heal_zone_ids: list[shapes.Circle]
    label: text.Label

    rendering_enabled: bool
    
    def __init__(self):
        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE)

        self.rendering_enabled = True
                
        pyglet.clock.schedule_interval(self.update, 1/60.0)
        self.window.push_handlers(self)


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self) -> None:
        self.window.clear()

        if self.rendering_enabled:
            self.curriculum.draw()
    

    def update(self, _dt: float):
        if not self.curriculum.update():
            pyglet.app.exit()
            return


    def run(self):
        self.curriculum = Curriculum()
        pyglet.app.run()