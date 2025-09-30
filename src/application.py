
import math
import threading
import pyglet
from pyglet import shapes, text

from config import GRID_SIZE, NUM_HEAL_ZONES, NUM_INDIVS, NUM_RAD_ZONES, WINDOW_SCALE
from src.curriculum import Curriculum

class Application:
    curriculum: Curriculum

    rendering_enabled: bool

    stats_document: text.document.UnformattedDocument
    layout: text.layout.TextLayout
    heal_zone_ids: list[shapes.Circle]
    rad_zone_ids: list[shapes.Circle]
    indiv_ids: list[shapes.Circle]
    label: text.Label


    def __init__(self):
        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE)

        self.rendering_enabled = True
                
        self.window.on_draw = self.on_draw
        self.window.on_key_press = self.on_key_press

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


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self) -> None:
        self.window.clear()
        sim = self.curriculum.sim

        if self.rendering_enabled and sim is not None:
            if sim.heal_zones is not None:
                for heal_zone in sim.heal_zones:
                        # TODO: Why are we just grabbing the first circle instance? If this works, maybe we should only 
                        # make a single instance altogether and use it as a brush. Same goes for rad_zones.
                        circle = self.heal_zone_ids[0]
                        circle.radius = heal_zone.radius * WINDOW_SCALE
                        circle.position = heal_zone.position[0] * WINDOW_SCALE, heal_zone.position[1] * WINDOW_SCALE
                        circle.draw()

            if sim.rad_zones is not None:
                for rad_zone in sim.rad_zones:
                    circle = self.rad_zone_ids[0]
                    circle.radius = rad_zone.radius * WINDOW_SCALE
                    circle.position = rad_zone.position[0] * WINDOW_SCALE, rad_zone.position[1] * WINDOW_SCALE
                    circle.draw()

            if sim.indiv_updates is not None:
                for i, update in enumerate(sim.indiv_updates):
                    percent_healed = min(abs(update.times_healed) / self.curriculum.theoretical_max_fitness, 1)

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
                    f"Avg. fitness: { round(self.curriculum.avg_times_healed, 2) }",
                    f"Moving avg. fitness: { round(self.curriculum.moving_avg_times_healed, 2) }",
                    f"Steps remaining: {sim.steps_remaining}",
                    f"Last run time: { round(self.curriculum.last_run_time, 2) }s",
                    f"Num simulations: { sim.indivs[0].model.num_simulations }"
                ]

                self.stats_document.text = '\n'.join(analytics)

            self.layout.draw()
    

    def run(self):
        self.curriculum = Curriculum()

        thread = threading.Thread(target=self.curriculum.run)
        thread.daemon = True
        thread.start()

        pyglet.app.run()

