from multiprocessing.connection import PipeConnection
import time
import tkinter as tk

from shared.lib import GRID_SIZE, NUM_INDIVS, WINDOW_SCALE, Individual

class Viewer:
    root: tk.Tk
    canvas: tk.Canvas
    rendering_enabled: bool
    pipe: PipeConnection
    indivs: list[Individual]
    ids: list[int]

    render_interval = 64 / 1000
    last_render = time.time()

    def __init__(self, pipe: PipeConnection):
        self.ids = [0] * NUM_INDIVS
        self.indivs: list[Individual] = [Individual(0, (0, 0))] * NUM_INDIVS
        self.pipe = pipe

        self.root = tk.Tk()
        self.root.title("Shared Memory Viewer")
        self.root.bind("<KeyPress>", self.on_key_press)

        self.canvas = tk.Canvas(self.root, width=GRID_SIZE * WINDOW_SCALE, height=GRID_SIZE * WINDOW_SCALE, borderwidth=0, highlightthickness=0, bg="black")
        self.canvas.grid()

        self.rendering_enabled = True
                
        for i, indiv in enumerate(self.indivs):
            id = self.canvas.create_oval(*get_circle_coords(indiv.position, WINDOW_SCALE), outline="white", fill="white")
            self.ids[i] = id

        running = True
        while running:
            running = self.update()

        self.root.after(16, self.render)
        self.root.mainloop()
        self.root.quit()


    def on_key_press(self, event: tk.Event): 
        if event.char == " ":
            self.rendering_enabled = not self.rendering_enabled

    def update(self):
        now = time.time()

        try:
            self.indivs = self.pipe.recv()
        except EOFError:
            self.root.quit()
            return False
        
        if now - self.last_render > self.render_interval:
            self.last_render = now
            self.render()

        self.root.update()

        return True


    def render(self):
        if self.rendering_enabled:
            for i, indiv in enumerate(self.indivs):
                self.canvas.coords(self.ids[i], *get_circle_coords(indiv.position, WINDOW_SCALE))

            self.root.update()
        self.root.after(16, self.render)
    

def get_circle_coords(pos: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    return pos[0] - radius, pos[1] + radius, pos[0] + radius, pos[1] - radius


def viewer_worker(pipe: PipeConnection):
    Viewer(pipe)