

from config import LOAD_MODELS, NUM_INDIVS
from src.individual import Individual

def load_individuals() -> list[Individual]:
    indivs = []

    for i in range(NUM_INDIVS):
        indiv = Individual()

        if LOAD_MODELS:
            try:
                indiv.model.inner.load_weights(f".models/{i}.h5")
            except Exception as e:
                print(f"Failed to load model, initializing randomly. Error: {e}")

        indivs.append(indiv)

    return indivs