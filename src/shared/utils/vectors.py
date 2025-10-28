import math

def normalize_vector(vector):
    dx, dy = vector
    magnitude = math.sqrt(dx**2 + dy**2)
    if magnitude != 0:
        return (dx / magnitude, dy / magnitude)
    else:
        return (0, 0)