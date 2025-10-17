import cProfile
from multiprocessing import Process
import pstats

from core.project import Project
from services.projects import load_projects
from core.application import Application
from config import PROFILER

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

def main() -> None:
    projects = load_projects()

    if not projects:
        print("No projects found in .projects. Exiting.")
        return

    print("\n")

    # Header
    print(f"{"#".ljust(3)}{"Project".ljust(40)}{"Strategy".ljust(20)}")

    # List projects
    for i, project_data in enumerate(projects, start=1):
        print(f"{str(i).ljust(3)}{str(project_data.id).ljust(40)}{str(project_data.strategy).ljust(20)}")

    print("\n")

    # Prompt for selection
    selected_index: int | None = None

    while selected_index is None:
        choice = input(f"Select a project [1-{len(projects)}] (or q to quit): ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            return
        try:
            idx = int(choice)
            if 1 <= idx <= len(projects):
                selected_index = idx - 1
            else:
                print(f"Please enter a number between 1 and {len(projects)}.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

    project = Project.from_data(projects[selected_index])
    app = Application(project)
    app.run()

if __name__ == "__main__":
    main()