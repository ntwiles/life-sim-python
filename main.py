import cProfile
from multiprocessing import Process
import pstats

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

    print("\n")

    print(f"{"#".ljust(3)}{"Project".ljust(40)}{"Curriculum".ljust(20)}")

    for i, project_data in enumerate(projects):
        print(f"{str(i + 1).ljust(3)}{str(project_data.id).ljust(40)}{project_data.curriculum.ljust(20)}")

    print("\n")

    project = Project.from_data(projects[0])
    app = Application(project)
    app.run()

if __name__ == "__main__":
    main()