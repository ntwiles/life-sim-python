from collections.abc import Callable
from functools import partial
import math
import pyglet
from pyglet import shapes, text
from shared.config import GRID_SIZE, HEAL_ZONE_RADIUS, WINDOW_SCALE, HEAL_ZONE_COUNT, RAD_ZONE_COUNT, NUM_INDIVS
from shared.types.drawing_data import SimulationDrawingData, ProjectDrawingData
from shared.types.individual_update_context import IndividualUpdateContext
from trainer.rendering.renderer import Renderer

GREEN_TRANS = (0, 255, 0, 50)
GREEN = (0, 255, 0, 255)
RED_TRANS = (255, 0, 0, 50)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
WHITE = (255, 255, 255, 255)

def _get_fitness_color( theoretical_max_fitness: float, indiv: IndividualUpdateContext,) -> tuple[int, int, int, int]:
    percent_healed = min(abs(indiv.times_healed) / theoretical_max_fitness, 1)

    r = 0
    g = 0
    b = 0

    if (indiv.times_healed > 0): 
        r = 255 - math.floor(percent_healed * 255)
        g = 255
        b = 255 - math.floor(percent_healed * 255)
    else: 
        r = 255
        g = 255 - math.floor(percent_healed * 255)
        b = 255 - math.floor(percent_healed * 255)

    return (r, g, b, 255)

def _get_heal_zone_dist_color(indiv: IndividualUpdateContext) -> tuple[int, int, int, int]:
    within_heal_zone = indiv.heal_zone_dist < HEAL_ZONE_RADIUS

    if within_heal_zone:
        return GREEN
    else:
        return RED
    
def _get_heal_zone_dir_color(indiv: IndividualUpdateContext) -> tuple[int, int, int, int]:
    dir_magnitude = math.sqrt(indiv.heal_zone_dir[0] ** 2 + indiv.heal_zone_dir[1] ** 2)

    if dir_magnitude == 0:
        return WHITE

    normalized_dir = (indiv.heal_zone_dir[0] / dir_magnitude, indiv.heal_zone_dir[1] / dir_magnitude)

    r = math.floor((normalized_dir[0] + 1) / 2 * 255)
    g = math.floor((normalized_dir[1] + 1) / 2 * 255)
    b = 255 - math.floor(((normalized_dir[0] + normalized_dir[1]) / 2 + 1) / 2 * 255)

    return (r, g, b, 255)

class PygletRenderer(Renderer):
    rendering_enabled: bool
    rendering_mode: int
    rendering_methods: list[Callable[[IndividualUpdateContext], tuple[int, int, int, int]]]

    project_stats_document: text.document.UnformattedDocument
    sim_stats_document: text.document.UnformattedDocument

    sim_stats_layout: text.layout.TextLayout
    heal_zone_ids: list[shapes.Circle]
    rad_zone_ids: list[shapes.Circle]
    indiv_ids: list[shapes.Circle]
    label: text.Label

    def __init__(self, theoretical_max_fitness: float):
        self.theoretical_max_fitness = theoretical_max_fitness
        self.rendering_enabled = True
        self.rendering_mode = 0
        self.latest_sim_data: SimulationDrawingData | None = None
        self.latest_project_data: ProjectDrawingData | None = None

        self.window = pyglet.window.Window(GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE) # type: ignore[abstract]
        self.window.on_draw = self.on_draw # type: ignore[assignment]
        self.window.on_key_press = self.on_key_press # type: ignore[assignment]

        self.indiv_ids = [shapes.Circle(0, 0, WINDOW_SCALE) for _ in range(NUM_INDIVS)]
        self.heal_zone_ids = [shapes.Circle(0, 0, WINDOW_SCALE, color=GREEN_TRANS, segments=32) 
                             for _ in range(HEAL_ZONE_COUNT)]
        self.rad_zone_ids = [shapes.Circle(0, 0, WINDOW_SCALE, color=RED_TRANS, segments=32)
                            for _ in range(RAD_ZONE_COUNT)]
        
        project_style = dict(
            margin_left="10px",
            margin_top="10px",
            font_name='Times New Roman',
            font_size=12,
            color=(255, 255, 255, 255)
        )
        
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

        self.rendering_methods = [
            partial(_get_fitness_color, self.theoretical_max_fitness),
            _get_heal_zone_dist_color,
            _get_heal_zone_dir_color
        ]

        self.latest_sim_data = None
        self.sim_stats_document = text.decode_text('')
        self.sim_stats_document.set_style(0, len(self.sim_stats_document.text), sim_style)
        self.sim_stats_layout = text.layout.TextLayout(self.sim_stats_document, GRID_SIZE * WINDOW_SCALE, GRID_SIZE * WINDOW_SCALE, multiline=True)
        

    def start(self) -> None:
        pyglet.app.run()

    def stop(self) -> None:
        pyglet.app.exit()

    def on_draw(self) -> None:
        self.window.clear()

        if not self.rendering_enabled:
            return
                
        if self.latest_sim_data is not None:
            self._draw_sim_stats()
            self._draw_sim_entities()

        if self.latest_project_data is not None: 
            self._draw_project_stats()

    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.rendering_enabled = not self.rendering_enabled
        elif symbol == pyglet.window.key.TAB:
            self.rendering_mode = (self.rendering_mode + 1) % len(self.rendering_methods)

    def update_simulation(self, data: SimulationDrawingData) -> None:
        self.latest_sim_data = data

    def update_project(self, data: ProjectDrawingData) -> None:
        self.latest_project_data = data

    def is_running(self) -> bool:
        return not pyglet.app.event_loop.has_exit
    
    def _draw_project_stats(self):
        project = self.latest_project_data
        
        project_stats = [
            "Project:",            
            f"Last avg. fitness: { round(project.avg_times_healed, 2) }",
            f"Last sim duration: { round(project.last_sim_duration, 2) }s",
            f"Last training duration: { round(project.last_training_duration, 2) }s",
            f"Current generation: { project.generation_num }",
            f"Theoretical max fitness: { round(self.theoretical_max_fitness, 2) }"
        ]
        
        self.project_stats_document.text = '\n'.join(project_stats)
        self.project_stats_layout.draw()

    def _get_entity_color(self, indiv: IndividualUpdateContext) -> tuple[int, int, int, int]:
        return self.rendering_methods[self.rendering_mode](indiv)

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
            circle = self.indiv_ids[i]
            circle.position = update.next_position[0] * WINDOW_SCALE, update.next_position[1] * WINDOW_SCALE
            circle.color = self._get_entity_color(update)
            circle.draw()

    def _draw_sim_stats(self):
        sim = self.latest_sim_data

        sim_stats = [
            "Simulation:",
            f"Steps remaining: {sim.steps_remaining}",
        ]

        self.sim_stats_document.text = '\n'.join(sim_stats)
        self.sim_stats_layout.draw()
