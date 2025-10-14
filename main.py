import cProfile
from multiprocessing import Process
import pstats

from src.curricula.evolutionary import apply_evolutionary_curriculum
from src.project import Project
from src.services.projects import load_projects
from src.application import Application
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

def main():
    projects = load_projects()
    project = Project.from_data(projects[0])
    app = Application(project)
    app.run()

if __name__ == "__main__":
    main()