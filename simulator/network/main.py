import tensorflow as tf

from shared.lib import INPUT_SIZE, Individual
from simulator.main import IndividualUpdateContext

def get_input_values(indiv: Individual, context: IndividualUpdateContext, generation_time: int) -> list[float]:
    prev_position_dir_x = indiv.position[0] - indiv.previous_position[0]
    prev_position_dir_y = indiv.position[1] - indiv.previous_position[1]

    heal_zone_angle = context.heal_zone_angle
    heal_zone_cos = tf.cos(heal_zone_angle)
    heal_zone_sin = tf.sin(heal_zone_angle)

    input_values = [
        prev_position_dir_x,
        prev_position_dir_y,
        generation_time,
        heal_zone_cos.numpy(),
        heal_zone_sin.numpy(),
    ]

    if len(input_values) != INPUT_SIZE:
        raise ValueError(f"Expected {INPUT_SIZE} inputs, got {len(input_values)}")
    
    return input_values

def decide(indiv: Individual, context: IndividualUpdateContext, generation_time: int) -> tuple[int, int]:
    output_values = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (0, 0),
    ]
            
    with tf.device('/GPU:0'):       
        input_values = get_input_values(indiv, context, generation_time)
        input_layer = tf.constant([input_values], dtype=tf.float32)

        output = indiv.model(input_layer)
        decision = tf.argmax(output, axis=1).numpy()[0]

    return output_values[decision]