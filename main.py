import cProfile
from multiprocessing import Process, Queue
import pstats

from shared.lib import PROFILER
from simulator.main import simulator_worker
from viewer.main import viewer_worker


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
    queue = Queue()

    viewer = make_process(viewer_worker, queue)
    viewer.start()

    simulator = make_process(simulator_worker, queue)
    simulator.start()

    simulator.join()
    viewer.join()

if __name__ == "__main__":
    main()