
import threading

from shared.types.drawing_data import SimulationDrawingData
from trainer.core.project import Project
from trainer.core.simulation_orchestrator import GenerationCompleteData, SimulationOrchestrator
from trainer.rendering.renderer import Renderer

class Application:
    project: Project
    renderer: Renderer

    def __init__(self, project: Project, renderer: Renderer):
        self.project = project      
        self.renderer = renderer
        self.orchestrator = SimulationOrchestrator(
            project_id=self.project.id,
            strategy_key=self.project.strategy_key
        )

    def run(self):
        def on_sim_update(data: SimulationDrawingData):
            self.renderer.update_simulation(data)

        def on_generation_complete(data: GenerationCompleteData):
            draw_data, is_training_complete = self.project.update_project(data)
            self.renderer.update_project(draw_data)

            if is_training_complete:
                self.renderer.stop()
                self.orchestrator.stop()

        thread = threading.Thread(target=self.orchestrator.run, args=(on_sim_update, on_generation_complete))
        thread.daemon = True
        thread.start()

        self.renderer.start()

