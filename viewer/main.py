from multiprocessing.connection import PipeConnection
import tkinter as tk

from shared.lib import Individual

ENTITY_RADIUS = 5

def get_circle_coords(pos: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    return pos[0] - radius, pos[1] + radius, pos[0] + radius, pos[1] - radius


def update(root: tk.Tk, canvas: tk.Canvas, pipe: PipeConnection, ids: list[int]):    
    try:
        indivs: list[Individual] = pipe.recv()

        for i, indiv in enumerate(indivs):
            canvas.coords(ids[i], *get_circle_coords(indiv.position, ENTITY_RADIUS))
    except EOFError:
        root.quit()
        return
    
    root.after(10, lambda: update(root, canvas, pipe, ids))


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
    
    root.after(10, lambda: update(root, canvas, pipe, ids))
    root.mainloop()

    root.quit()



