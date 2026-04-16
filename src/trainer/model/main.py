from dataclasses import dataclass
import tensorflow as tf

from shared.config import INPUT_SIZE, MUTATION_MAGNITUDE, MUTATION_RATE

@dataclass
class Model:
    inner: tf.keras.Sequential


def create_model() -> Model:
    inner = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(INPUT_SIZE,)),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

    return Model(inner=inner)


def clone_model(parent: Model) -> Model:
    inner = tf.keras.models.clone_model(parent.inner)
    inner.set_weights(parent.inner.get_weights())
    return Model(inner=inner)


def clone_and_mutate_model(parent: Model) -> Model:
    inner = tf.keras.models.clone_model(parent.inner)
    inner.set_weights(parent.inner.get_weights())
    inner = mutate_weights(inner)
    return Model(inner=inner)


def mutate_weights(inner: tf.keras.Sequential) -> tf.keras.Sequential:
    for var in inner.trainable_variables:
        mutation_mask = tf.random.uniform(var.shape) < MUTATION_RATE
        random_mutations = tf.random.normal(var.shape, mean=0.0, stddev=MUTATION_MAGNITUDE)
        var.assign(tf.where(mutation_mask, var + random_mutations, var))

    return inner