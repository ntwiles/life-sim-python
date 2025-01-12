import numpy as np
import tensorflow as tf

from config import INPUT_SIZE
from simulator.individual import Individual
from simulator.main import IndividualUpdateContext

def get_input_values(indiv: Individual, context: IndividualUpdateContext, t: float) -> list[float]:
    prev_position_dir_x = indiv.position[0] - indiv.previous_position[0]
    prev_position_dir_y = indiv.position[1] - indiv.previous_position[1]

    heal_zone_dir_x, heal_zone_dir_y = context.heal_zone_dir

    input_values = [
        prev_position_dir_x,
        prev_position_dir_y,
        context.heal_zone_dist,
        heal_zone_dir_x,
        heal_zone_dir_y,
    ]

    if len(input_values) != INPUT_SIZE:
        raise ValueError(f"Expected {INPUT_SIZE} inputs, got {len(input_values)}")
    
    return input_values

def decide(indiv: Individual, context: IndividualUpdateContext, t: float) -> tuple[int, int]:
    output_values = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (0, 0),
    ]
               
    input_values = np.array([get_input_values(indiv, context, t)])

    output = indiv.model(input_values)
    decision = tf.argmax(output, axis=1).numpy()[0]

    return output_values[decision]