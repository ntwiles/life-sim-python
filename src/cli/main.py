import cProfile
from multiprocessing import Process
import os
import pstats

from trainer.core.project import Project
from trainer.core.application import Application
from shared.config import NUM_INDIVS, PROFILER
from shared.services.individuals import save_individuals
from shared.services.projects import load_project_datas, save_project_data, save_selected_project_id
from shared.types.strategies import StrategyKey
from trainer.simulation.individual import Individual
from trainer.rendering.pyglet_renderer import PygletRenderer


def run_with_profiling(target, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    target(*args, **kwargs)
    profiler.disable()

    filename = f"{target.__name__}_profile.txt"

    with open(filename, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("time")
        stats.print_stats()

    print(f"Profiling results saved to {filename}")


def make_process(target, *args, **kwargs):
    if PROFILER:
        return Process(target=run_with_profiling, args=(target, *args), kwargs=kwargs)
    else:
        return Process(target=target, args=args, kwargs=kwargs)


def create_new_project(strategy_key: StrategyKey) -> Project:
    project = Project.new(strategy_key)
    print(f"Creating new project with id {project.id} and strategy {project.strategy_key}...")
    os.makedirs(f".data/projects/{project.id}", exist_ok=True)
    save_project_data(project.to_data())
    save_individuals(project.id, [Individual().to_data() for _ in range(NUM_INDIVS)])
    return project


def main() -> None:
    projects = load_project_datas()

    if not projects:
        print("No projects found in .data/projects. Exiting.")
        return

    print("\n")

    # Header
    print(f"{"#".ljust(3)}{"Project".ljust(40)}{"Strategy".ljust(20)}{"Generations".ljust(15)}")

    # List projects
    for i, project_data in enumerate(projects, start=1):
        print(f"{str(i).ljust(3)}{str(project_data.id).ljust(40)}{str(project_data.strategy).ljust(20)}{str(project_data.generation_num).ljust(15)}")

    print("\n")

    # Prompt for selection
    selected_index: int | None = None

    while selected_index is None:
        choice = input(f"Select a project [1-{len(projects)}] (or q to quit, n for new project): ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            return
        if choice.lower() == "n":
            new_project = create_new_project("evolutionary")
            print(f"\nCreated project {new_project.id}")
            projects.append(new_project.to_data())
            selected_index = len(projects) - 1
        try:
            idx = int(choice)
            if 1 <= idx <= len(projects):
                selected_index = idx - 1
            else:
                print(f"Please enter a number between 1 and {len(projects)}.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

    selected_project_data = projects[selected_index]
    save_selected_project_id(selected_project_data.id)
    project = Project.from_data(selected_project_data)
    renderer = PygletRenderer(project.theoretical_max_fitness)

    app = Application(project, renderer)
    app.run()
