
import math
import threading
import pyglet
from pyglet import shapes, text

from config import GRID_SIZE, NUM_HEAL_ZONES, NUM_INDIVS, NUM_RAD_ZONES, WINDOW_SCALE
from src.drawing_data import SimulationDrawingData, ProjectDrawingData
from src.project import Project

class Application:
    project: Project

    rendering_enabled: bool

    stats_document: text.document.UnformattedDocument
    layout: text.layout.TextLayout
    heal_zone_ids: list[shapes.Circle]
    rad_zone_ids: list[shapes.Circle]
    indiv_ids: list[shapes.Circle]
    label: text.Label

    latest_sim_data: SimulationDrawingData | None
    latest_project_data: ProjectDrawingData | None


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

        self.latest_sim_data = None
        self.latest_project_data = None


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self) -> None:
        self.window.clear()

        if not self.rendering_enabled:
            return
        
        analytics = []
        
        if self.latest_sim_data is not None:
            sim = self.latest_sim_data

            for heal_zone in sim.heal_zones:
                    # TODO: Why are we just grabbing the first circle instance? If this works, maybe we should only 
                    # make a single instance altogether and use it as a brush. Same goes for rad_zones.
                    circle = self.heal_zone_ids[0]
                    circle.radius = heal_zone.radius * WINDOW_SCALE
                    circle.position = heal_zone.position[0] * WINDOW_SCALE, heal_zone.position[1] * WINDOW_SCALE
                    circle.draw()

            for rad_zone in sim.rad_zones:
                circle = self.rad_zone_ids[0]
                circle.radius = rad_zone.radius * WINDOW_SCALE
                circle.position = rad_zone.position[0] * WINDOW_SCALE, rad_zone.position[1] * WINDOW_SCALE
                circle.draw()

            for i, update in enumerate(sim.indiv_updates):
                percent_healed = min(abs(update.times_healed) / self.project.theoretical_max_fitness, 1)

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
                f"Avg. fitness: { round(self.project.avg_times_healed, 2) }",
                f"Moving avg. fitness: { round(self.project.moving_avg_times_healed, 2) }",
                f"Steps remaining: {sim.steps_remaining}",
                f"Model generations: { sim.model_num_generations }"
            ]

        if self.latest_project_data is not None: 
            project = self.latest_project_data

            analytics.append(f"Last sim duration: { round(project.last_sim_duration, 2) }s")
            analytics.append(f"Last training duration: { round(project.last_training_duration, 2) }s")

        self.stats_document.text = '\n'.join(analytics)
        self.layout.draw()
    

    def run(self):
        self.project = Project()

        def handle_sim_updates(data: SimulationDrawingData):
            self.latest_sim_data = data

        def handle_project_updates(data: ProjectDrawingData):
            self.latest_project_data = data

        thread = threading.Thread(target=self.project.run, args=(handle_sim_updates, handle_project_updates))
        thread.daemon = True
        thread.start()

        pyglet.app.run()

