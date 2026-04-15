
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from collections import deque

from shared.config import PLOT_MAX_DATA_POINTS
from shared.services.projects import ProjectData, load_project_data

EMA_ALPHA_100 = 2 / (100 + 1)
EMA_ALPHA_1000 = 2 / (1000 + 1) 
EMA_ALPHA_10000 = 2 / (10000 + 1)

MAJOR_MINOR_GRID_ALPHA = 0.6

def _calculate_exponential_moving_average(a: float, new_value: float, previous_ema: float | None) -> float:
    if previous_ema is None:
        return new_value
    
    return (new_value * a) + (previous_ema * (1 - a))

def initialize_plot():
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
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=MAJOR_MINOR_GRID_ALPHA)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=MAJOR_MINOR_GRID_ALPHA)

    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    # Create plot lines
    moving_avg_fitness_100_line, = ax.plot([], [], lw=2, label='Moving avg fitness (100 gen)')
    moving_avg_fitness_1000_line, = ax.plot([], [], lw=2, label='Moving avg fitness (1000 gen)')
    moving_avg_fitness_10000_line, = ax.plot([], [], lw=2, label='Moving avg fitness (10000 gen)')

    # Setup legend with styling
    legend = ax.legend(loc='upper left')
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('gray')
    for text in legend.get_texts():
        text.set_color('white')

    return fig, ax, moving_avg_fitness_100_line, moving_avg_fitness_1000_line, moving_avg_fitness_10000_line

def calculate_initial_last_generation(project: ProjectData) -> int:
    available_generations = len(project.last_k_avg_fitness)
    return project.generation_num - available_generations if available_generations > 0 else project.generation_num

def main(project_id: str) -> None:
    fig, ax, moving_avg_fitness_100_line, moving_avg_fitness_1000_line, moving_avg_fitness_10000_line = initialize_plot()

    xs: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)

    moving_avg_fitness_100_ys: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)
    moving_avg_fitness_1000_ys: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)
    moving_avg_fitness_10000_ys: deque[float] = deque(maxlen=PLOT_MAX_DATA_POINTS)

    project = load_project_data(project_id)

    last_generation = calculate_initial_last_generation(project)
    last_ema_100 = None
    last_ema_1000 = None
    last_ema_10000 = None

    def handle_generation(project: ProjectData, generation: int):
        nonlocal xs
        nonlocal moving_avg_fitness_100_ys
        nonlocal moving_avg_fitness_1000_ys
        nonlocal moving_avg_fitness_10000_ys
        nonlocal last_ema_100
        nonlocal last_ema_1000
        nonlocal last_ema_10000

        generation_index = generation - (project.generation_num - len(project.last_k_avg_fitness)) - 1

        last_avg_fitness = project.last_k_avg_fitness[generation_index]

        xs.append(generation)

        moving_avg_fitness_100 = _calculate_exponential_moving_average(EMA_ALPHA_100, last_avg_fitness, last_ema_100)
        moving_avg_fitness_100_ys.append(moving_avg_fitness_100)
        moving_avg_fitness_100_line.set_data(xs, moving_avg_fitness_100_ys)
        last_ema_100 = moving_avg_fitness_100

        moving_avg_fitness_1000 = _calculate_exponential_moving_average(EMA_ALPHA_1000, last_avg_fitness, last_ema_1000)
        moving_avg_fitness_1000_ys.append(moving_avg_fitness_1000)
        moving_avg_fitness_1000_line.set_data(xs, moving_avg_fitness_1000_ys)
        last_ema_1000 = moving_avg_fitness_1000

        moving_avg_fitness_10000 = _calculate_exponential_moving_average(EMA_ALPHA_10000, last_avg_fitness, last_ema_10000)
        moving_avg_fitness_10000_ys.append(moving_avg_fitness_10000)
        moving_avg_fitness_10000_line.set_data(xs, moving_avg_fitness_10000_ys)
        last_ema_10000 = moving_avg_fitness_10000


        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()


    def on_timer():
        nonlocal last_generation

        project = load_project_data(project_id)

        if project.generation_num <= last_generation:
            return
        
        num_generations_to_catch_up = project.generation_num - last_generation
        for _ in range(num_generations_to_catch_up):
            handle_generation(project, last_generation + 1)
            last_generation += 1

    timer = fig.canvas.new_timer(interval=60_000)  # 60 seconds

    timer.add_callback(on_timer)
    timer.start()

    on_timer()

    plt.ioff()
    plt.show()