from multiprocessing import Event
from multiprocessing.connection import PipeConnection
from multiprocessing.shared_memory import SharedMemory
import struct

import numpy as np
import tkinter as tk

ENTITY_RADIUS = 5

def get_circle_coords(pos: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    return pos[0] - radius, pos[1] + radius, pos[0] + radius, pos[1] - radius


def update(root: tk.Tk, canvas: tk.Canvas, pipe: PipeConnection, ids: list[int]):    
    try:
        positions = pipe.recv()

        for i, pos in enumerate(positions):
            canvas.coords(ids[i], *get_circle_coords(pos, ENTITY_RADIUS))
    except EOFError:
        root.quit()
        return
    
    root.after(10, lambda: update(root, canvas, pipe, ids))


def viewer_worker(pipe: PipeConnection):
    root = tk.Tk()
    root.title("Shared Memory Viewer")

    canvas = tk.Canvas(root, width=1000, height=1000, borderwidth=0, highlightthickness=0, bg="black")
    canvas.grid()

    ids = [0] * 10
    positions = [(0, 0)] * 10

    for i, pos in enumerate(positions):
        id = canvas.create_oval(*get_circle_coords(pos, ENTITY_RADIUS), outline="white", fill="white")
        ids[i] = id
    
    root.after(10, lambda: update(root, canvas, pipe, ids))
    root.mainloop()

    root.quit()



