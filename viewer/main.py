from multiprocessing.connection import PipeConnection
import tkinter as tk

from shared.lib import Individual

ENTITY_RADIUS = 5

def get_circle_coords(pos: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    return pos[0] - radius, pos[1] + radius, pos[0] + radius, pos[1] - radius


def viewer_worker(pipe: PipeConnection):
    root = tk.Tk()
    root.title("Shared Memory Viewer")

    canvas = tk.Canvas(root, width=1000, height=1000, borderwidth=0, highlightthickness=0, bg="black")
    canvas.grid()

    ids = [0] * 3000
    indivs: list[Individual] = [Individual(0, (0, 0))] * 3000

    for i, indiv in enumerate(indivs):
        id = canvas.create_oval(*get_circle_coords(indiv.position, ENTITY_RADIUS), outline="white", fill="white")
        ids[i] = id

    while True:
        try:
            indivs = pipe.recv()

            for i, indiv in enumerate(indivs):
                canvas.coords(ids[i], *get_circle_coords(indiv.position, ENTITY_RADIUS))

            root.update()
        except EOFError:
            root.quit()
            break
    
    root.quit()
    print("Viewer done")



