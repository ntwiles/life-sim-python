
import math
from queue import Empty, Queue
import threading
import pyglet
from pyglet import shapes, text

from config import GRID_SIZE, HEAL_ZONE_COUNT, NUM_INDIVS, RAD_ZONE_COUNT, WINDOW_SCALE
from visualization.plotting import plot_realtime_metrics
from visualization.drawing_data import SimulationDrawingData, ProjectDrawingData
from core.project import Project

class Application:
    project: Project

    rendering_enabled: bool

    project_stats_document: text.document.UnformattedDocument
    sim_stats_document: text.document.UnformattedDocument

    sim_stats_layout: text.layout.TextLayout
    heal_zone_ids: list[shapes.Circle]
    rad_zone_ids: list[shapes.Circle]
    indiv_ids: list[shapes.Circle]
    label: text.Label

    latest_sim_data: SimulationDrawingData | None
    latest_project_data: ProjectDrawingData | None

    plot_queue: Queue[ProjectDrawingData]


    def __init__(self, project: Project):
        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE) # type: ignore[abstract]
        self.window.on_draw = self.on_draw # type: ignore[assignment]
        self.window.on_key_press = self.on_key_press # type: ignore[assignment]

        self.project = project      
        self.rendering_enabled = True

        self.indiv_ids = []
        for _ in range(NUM_INDIVS):
            self.indiv_ids.append(shapes.Circle(0,0, WINDOW_SCALE))

        self.heal_zone_ids = []
        for _ in range(HEAL_ZONE_COUNT):
            self.heal_zone_ids.append(shapes.Circle(0, 0, WINDOW_SCALE, color=(110, 255, 100, 60), segments=32))

        self.rad_zone_ids = []
        for _ in range(RAD_ZONE_COUNT):
            self.rad_zone_ids.append(shapes.Circle(0, 0, WINDOW_SCALE, color=(255, 100, 100, 60), segments=32))
            
        project_style = dict(
            margin_left="10px",
            margin_top="10px",
            font_name='Times New Roman',
            font_size=12,
            color=(255, 255, 255, 255)
        )

        self.latest_project_data = None
        self.project_stats_document = text.decode_text('')
        self.project_stats_document.set_style(0, len(self.project_stats_document.text), project_style)
        self.project_stats_layout = text.layout.TextLayout(self.project_stats_document, GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE, multiline=True)

        sim_style = dict(
            margin_right="10px",
            margin_top="10px",
            font_name='Times New Roman',
            font_size=12,
            color=(255, 255, 255, 255),
            align='right'
        )

        self.latest_sim_data = None
        self.sim_stats_document = text.decode_text('')
        self.sim_stats_document.set_style(0, len(self.sim_stats_document.text), sim_style)
        self.sim_stats_layout = text.layout.TextLayout(self.sim_stats_document, GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE, multiline=True)

        self.plot_queue = Queue(maxsize=1)


    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled


    def on_draw(self) -> None:
        self.window.clear()

        if not self.rendering_enabled:
            return
                
        if self.latest_sim_data is not None:
            self._draw_sim_stats()
            self._draw_sim_entities()

        if self.latest_project_data is not None: 
            self._draw_project_stats()


    def run(self):
        def handle_sim_updates(data: SimulationDrawingData):
            self.latest_sim_data = data

        def handle_project_updates(data: ProjectDrawingData):
            self.latest_project_data = data

            while True:
                try:
                    self.plot_queue.get_nowait()
                except Empty:
                    break

            self.plot_queue.put_nowait(data)

        thread = threading.Thread(target=self.project.run, args=(handle_sim_updates, handle_project_updates))
        thread.daemon = True
        thread.start()

        thread = threading.Thread(target=plot_realtime_metrics, args=(self.plot_queue,))
        thread.daemon = True
        thread.start()

        pyglet.app.run()


    def _draw_project_stats(self):
        project = self.latest_project_data
        
        project_stats = [
            "Project:",            
            f"Last avg. fitness: { round(project.avg_times_healed, 2) }",
            f"Moving avg. fitness: { round(project.moving_avg_times_healed, 2) }",
            f"Last sim duration: { round(project.last_sim_duration, 2) }s",
            f"Last training duration: { round(project.last_training_duration, 2) }s",
            f"Theoretical max fitness: { round(self.project.theoretical_max_fitness, 2) }"
        ]
        
        self.project_stats_document.text = '\n'.join(project_stats)
        self.project_stats_layout.draw()

    def _draw_sim_entities(self):
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

    def _draw_sim_stats(self):
        sim = self.latest_sim_data

        sim_stats = [
            "Simulation:",
            f"Steps remaining: {sim.steps_remaining}",
            f"Model generations: { sim.model_num_generations }"
        ]

        self.sim_stats_document.text = '\n'.join(sim_stats)
        self.sim_stats_layout.draw()
