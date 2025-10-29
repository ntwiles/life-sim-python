
import threading
import pyglet

from shared.types.drawing_data import SimulationDrawingData, ProjectDrawingData
from trainer.core.project import Project
from trainer.rendering.renderer import Renderer

class Application:
    project: Project
    renderer: Renderer

    def __init__(self, project: Project, renderer: Renderer):
        self.project = project      
        self.renderer = renderer

    def run(self):
        def handle_sim_updates(data: SimulationDrawingData):
            self.renderer.update_simulation(data)

        def handle_project_updates(data: ProjectDrawingData):
            self.renderer.update_project(data)

        thread = threading.Thread(target=self.project.run, args=(handle_sim_updates, handle_project_updates))
        thread.daemon = True
        thread.start()

        pyglet.app.run()


