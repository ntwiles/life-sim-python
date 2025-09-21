from dataclasses import dataclass
import tensorflow as tf

from config import INPUT_SIZE, MUTATION_MAGNITUDE, MUTATION_RATE, GATE_DISABLE_RATE

@dataclass
class Model:
    inner: tf.keras.Sequential
    num_simulations: int


def create_model() -> Model:
    inner = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(INPUT_SIZE,)),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
    
    return Model(inner=inner, num_simulations=0)


def mutate_weights(weights: tf.keras.Sequential):
    for var in weights.inner.trainable_variables:
        mutation_mask = tf.random.uniform(var.shape) < MUTATION_RATE
        random_mutations = tf.random.normal(var.shape, mean=0.0, stddev=MUTATION_MAGNITUDE)
        var.assign(tf.where(mutation_mask, var + random_mutations, var))

def mutate_weights_with_gating(weights: tf.keras.Sequential):
    mutate_weights(weights)

    for var in weights.inner.trainable_variables:
        disable_mask = tf.random.uniform(var.shape) < GATE_DISABLE_RATE
        var.assign(tf.where(disable_mask, 0.0, var))