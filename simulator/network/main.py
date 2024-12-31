import math
import tensorflow as tf

from shared.lib import Individual
from simulator.main import IndividualUpdateContext

def decide(indiv: Individual, context: IndividualUpdateContext, generation_time: int) -> tuple[int, int]:
    prev_position_dir_x = indiv.position[0] - indiv.previous_position[0]
    prev_position_dir_y = indiv.position[1] - indiv.previous_position[1]

    food_angle = context.food_angle

    with tf.device('/GPU:0'):
        input_values = [
            prev_position_dir_x,
            prev_position_dir_y,
            generation_time,
            math.cos(food_angle),
            math.sin(food_angle),
        ]

        input_layer = tf.constant([input_values], dtype=tf.float32)  # Batch size of 1

        output = indiv.model(input_layer)
        decision = tf.argmax(output, axis=1).numpy()[0]

        output_values = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (0, 0),
        ]

    return output_values[decision]