from multiprocessing.connection import PipeConnection
from time import sleep

def simulator_worker(pipe: PipeConnection):
    positions = [
        (0, 0),
        (100, 100),
        (200, 200),
        (300, 300),
        (400, 400),
        (500, 500),
        (600, 600),
        (700, 700),
        (800, 800),
        (900, 900),
    ]

    steps = 1000

    while steps > 0:
        for i, pos in enumerate(positions):
            positions[i] = pos[0] + 1, pos[1] + 1

        sleep(.01)
        pipe.send(positions)

        steps -= 1

    pipe.close()


