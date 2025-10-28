
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from collections import deque

from core.config import PLOT_MAX_DATA_POINTS
from shared.services.projects import ProjectData, load_project_datas


def main() -> None:
    plt.ion() 
    plt.rcParams['toolbar'] = 'None'

    fig, ax = plt.subplots()
    fig.set_facecolor('black')

    ax.set_facecolor('black')
    ax.set_title('Fitness Over Generations', color='white')
    ax.set_xlabel('Generation', color='white')
    ax.set_ylabel('Average Fitness', color='white')

    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    moving_avg_fitness_100_line, = ax.plot([], [], lw=2, label='Moving avg fitness (100 gen)')
    moving_avg_fitness_1000_line, = ax.plot([], [], lw=2, label='Moving avg fitness (1000 gen)')
    moving_avg_fitness_10000_line, = ax.plot([], [], lw=2, label='Moving avg fitness (10000 gen)')

    legend = ax.legend(loc='upper left')
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('gray')
    for text in legend.get_texts():
        text.set_color('white')

    xs: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)
    
    moving_avg_fitness_100_ys: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)
    moving_avg_fitness_1000_ys: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)
    moving_avg_fitness_10000_ys: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)

    last_generation: int = -1

    def handle_generation(project: ProjectData, generation: int):
        prefix = list(project.last_k_avg_fitness)[:generation]
        last_10000_avg_fitness = prefix[-10000:]
        last_1000_avg_fitness = list(last_10000_avg_fitness)[-1000:]
        last_100_avg_fitness = list(last_10000_avg_fitness)[-100:]

        ready_100 = len(last_100_avg_fitness) == 100
        ready_1000 = len(last_1000_avg_fitness) == 1000
        ready_10000 = len(last_10000_avg_fitness) == 10000

        if not (ready_100 or ready_1000 or ready_10000):
            return
        
        xs.append(float(generation))

        if ready_100:
            moving_avg_fitness_100 = sum(last_100_avg_fitness) / len(last_100_avg_fitness)
            moving_avg_fitness_100_ys.append(moving_avg_fitness_100)
            moving_avg_fitness_100_line.set_data(xs, moving_avg_fitness_100_ys)

        if ready_1000:
            moving_avg_fitness_1000 = sum(last_1000_avg_fitness) / len(last_1000_avg_fitness)
            moving_avg_fitness_1000_ys.append(moving_avg_fitness_1000)

            k = len(moving_avg_fitness_1000_ys)
            x_tail = list(xs)[-k:]
            moving_avg_fitness_1000_line.set_data(x_tail, moving_avg_fitness_1000_ys)

        if ready_10000:
            moving_avg_fitness_10000 = sum(last_10000_avg_fitness) / len(last_10000_avg_fitness)
            moving_avg_fitness_10000_ys.append(moving_avg_fitness_10000)

            k = len(moving_avg_fitness_10000_ys)
            x_tail = list(xs)[-k:]
            moving_avg_fitness_10000_line.set_data(x_tail, moving_avg_fitness_10000_ys)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()


    def on_timer():
        nonlocal last_generation

        # TODO: Handle multiple projects
        project = load_project_datas()[0]

        if project.generation_num <= last_generation:
            return
        
        # TODO: We can't necessarily go back to the first generation, we only have PLOT_MAX_DATA_POINTS generations stored.
        num_generations_to_catch_up = project.generation_num - last_generation
        for _ in range(num_generations_to_catch_up):
            handle_generation(project, last_generation + 1)
            last_generation += 1

    timer = fig.canvas.new_timer(interval=60_000)  # 1 frame per minute

    timer.add_callback(on_timer)
    timer.start()

    plt.ioff()
    plt.show()

    if __name__ == "__main__":
        main()