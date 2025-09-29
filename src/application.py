
import threading
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
                
        self.window.on_draw = self.on_draw
        self.window.on_key_press = self.on_key_press


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self) -> None:
        self.window.clear()

        if self.rendering_enabled:
            self.curriculum.draw()
    

    def run(self):
        self.curriculum = Curriculum()

        thread = threading.Thread(target=self.curriculum.run)
        thread.daemon = True
        thread.start()

        pyglet.app.run()

