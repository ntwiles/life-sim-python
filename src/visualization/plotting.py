from collections import deque
from queue import Empty, Queue
import matplotlib.pyplot as plt

from core.config import PLOT_MAX_DATA_POINTS
from visualization.drawing_data import ProjectDrawingData


def plot_realtime_metrics(queue: Queue[ProjectDrawingData]):
    plt.ion() 
    plt.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_title('Fitness Over Generations', color='white')
    ax.set_xlabel('Generation', color='white')
    ax.set_ylabel('Average Times Healed', color='white')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    last_fitness_line, = ax.plot([], [], lw=2)
    avg_fitness_line, = ax.plot([], [], lw=2)

    xs: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)
    last_fitness_ys: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)
    avg_fitness_ys: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)

    def on_timer():
        updated = False
        while True:
            try:
                msg = queue.get_nowait()
            except Empty:
                break

            updated = True
            xs.append(len(xs))
            last_fitness_ys.append(msg.avg_times_healed)
            avg_fitness_ys.append(msg.moving_avg_times_healed)
        if updated:
            last_fitness_line.set_data(xs, last_fitness_ys)
            avg_fitness_line.set_data(xs, avg_fitness_ys)

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=500)  # ~2 Hz, easy on CPU/GPU
    timer.add_callback(on_timer)
    timer.start()

    plt.ioff()
    plt.show() 