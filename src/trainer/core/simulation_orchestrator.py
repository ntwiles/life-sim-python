
from dataclasses import dataclass
from threading import Event
import time
from typing import Callable
import uuid

from shared.config import LOAD_MODELS, NUM_INDIVS, SIM_STEPS
from shared.services.individuals import load_individual_data
from shared.types.drawing_data import  SimulationDrawingData

from trainer.simulation.individual import Individual
from trainer.simulation.main import Simulation
from trainer.strategies.main import strategy_functions

@dataclass
class GenerationCompleteData:
    last_sim_duration: float
    last_training_duration: float
    generation: list[Individual]


class SimulationOrchestrator:
    project_id: uuid.UUID
    apply_strategy: Callable[[list[Individual]], list[Individual]]
    _stop_event: Event


    def __init__(self, project_id: uuid.UUID, strategy_key: str):
        self.project_id = project_id
        self.apply_strategy = strategy_functions[strategy_key]
        self._stop_event = Event()

    def spawn_initial_generation(self) -> list[Individual]:
        if LOAD_MODELS:
            return [Individual.from_data(load_individual_data(self.project_id, i)) for i in range(NUM_INDIVS)]
        else:
            return [Individual() for _ in range(NUM_INDIVS)]

    def run(self, on_sim_update: Callable[[SimulationDrawingData], None], on_generation_complete: Callable[[GenerationCompleteData], None]):
        generation = self.spawn_initial_generation()

        while not self._stop_event.is_set():
            sim_time_started = time.time()
            self.sim = Simulation(generation, on_update=on_sim_update)
            self.sim.run(SIM_STEPS)
            sim_duration = time.time() - sim_time_started

            training_time_started = time.time() 
            next_generation = self.apply_strategy(generation)
            training_duration = time.time() - training_time_started

            on_generation_complete(GenerationCompleteData(
                last_sim_duration=sim_duration,
                last_training_duration=training_duration,
                generation=generation
            ))

            generation = next_generation

    def stop(self):
        self._stop_event.set()