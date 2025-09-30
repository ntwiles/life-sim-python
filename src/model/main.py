from dataclasses import dataclass
import tensorflow as tf

from config import INPUT_SIZE, MUTATION_MAGNITUDE, MUTATION_RATE, GATE_DISABLE_RATE

@dataclass
class Model:
    inner: tf.keras.Sequential
    num_simulations: int
    gate_masks: list[tf.Variable]


def create_model() -> Model:
    inner = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(INPUT_SIZE,)),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
    
    gate_masks = []

    for layer in inner.layers:
        if hasattr(layer, 'kernel'): 
            gate_mask = tf.Variable(
                tf.ones_like(layer.kernel), 
                trainable=False, 
                name=f"{layer.name}_gate_mask"
            )

            gate_masks.append(gate_mask)
        
        if hasattr(layer, 'bias') and layer.use_bias:
            bias_gate_mask = tf.Variable(
                tf.ones_like(layer.bias), 
                trainable=False,
                name=f"{layer.name}_bias_gate_mask"
            )

            gate_masks.append(bias_gate_mask)
    
    return Model(inner=inner, num_simulations=0, gate_masks=gate_masks)

def clone_model(parent: Model) -> Model:
    inner = tf.keras.models.clone_model(parent.inner)
    inner.set_weights(parent.inner.get_weights())

    gate_masks = [tf.Variable(tf.ones_like(gate_mask), trainable=False) for gate_mask in parent.gate_masks]

    return Model(inner=inner, num_simulations=parent.num_simulations, gate_masks=gate_masks)


# TODO: Mutate 
def clone_and_mutate_model(parent: Model, is_gating_enabled: bool) -> Model:
    inner = tf.keras.models.clone_model(parent.inner)
    inner.set_weights(parent.inner.get_weights()) 
    inner = mutate_weights(inner)

    gate_masks = [tf.Variable(tf.ones_like(gate_mask), trainable=False) for gate_mask in parent.gate_masks]
    model = Model(inner=inner, num_simulations=parent.num_simulations, gate_masks=gate_masks)

    return apply_gating(model) if is_gating_enabled else model


def mutate_weights(inner: tf.keras.Sequential) -> tf.keras.Sequential:
    for var in inner.trainable_variables:
        mutation_mask = tf.random.uniform(var.shape) < MUTATION_RATE
        random_mutations = tf.random.normal(var.shape, mean=0.0, stddev=MUTATION_MAGNITUDE)
        var.assign(tf.where(mutation_mask, var + random_mutations, var))
    
    return inner

def apply_gating(model: Model) -> Model:
    for gate_mask in model.gate_masks:
        disable_mask = tf.random.uniform(gate_mask.shape) < GATE_DISABLE_RATE
        gate_mask.assign(tf.where(disable_mask, 0.0, 1.0))

    return model