# Life Simulation with Evolutionary Neural Networks

A Python-based artificial life simulation where neural network "individuals" evolve through natural selection to navigate a 2D environment with beneficial heal zones and harmful radiation zones. Models are trained only through random mutation and natural selection.

## Overview

This project explores evolutionary algorithms applied to neural networks in a spatial survival simulation. Individual agents must learn to:

- Navigate toward healing zones for fitness benefits
- Avoid moving radiation zones that cause damage
- Survive and reproduce based on their accumulated healing score

## Architecture

### Neural Network Structure

- **Input Layer**: 10 features including:
  - Previous movement direction (2)
  - Nearest heal zone direction and distance (3)
  - Nearest radiation zone direction and distance (3)
  - Radiation zone movement direction (2)
- **Hidden Layers**: 12 → 8 neurons with ReLU activation
- **Output Layer**: 5 movement actions (N/S/E/W/Stay) with softmax

### Genetic Gating System

A custom structural mutation mechanism where:

- Binary gate masks control which neural connections are active
- Gates can disable individual weights without destroying learned values
- Provides architectural diversity while preserving genetic material
- Mutation rate: 0.3% of connections per generation

### Environment

- **Grid Size**: 300×300 simulation space
- **Heal Zones**: 7 stationary beneficial areas (radius 30)
- **Radiation Zones**: 6 moving harmful areas (radius 25)
- **Population**: 120 individuals per generation
- **Simulation Length**: 300 steps per generation

## Configuration

Key parameters in `config.py`:

```python
NUM_INDIVS = 120          # Population size
MUTATION_RATE = 0.01      # 1% of weights mutated per generation
MUTATION_MAGNITUDE = 0.001 # Standard deviation of weight changes
GATE_DISABLE_RATE = 0.003 # Probability of connection gating
SELECTION_RATE = 1/3      # Top 33% selected for breeding
```

### Selection Rate Constraints

The `SELECTION_RATE` must satisfy mathematical constraints to ensure proper population replacement:

- `NUM_INDIVS` must be evenly divisible by `SELECTION_RATE`
- The inverse of `SELECTION_RATE` (offspring per parent) must be an integer
- Valid examples: `1/2` (50%), `1/3` (33%), `1/4` (25%), `1/5` (20%)
- Invalid examples: `0.4` (40%), `1/7` (14.3%), `0.15` (15%)

## Evolution Mechanics

1. **Selection**: Fitness-proportional selection based on `times_healed` score
2. **Breeding**: Each selected parent produces 3 offspring
3. **Mutation**: Gaussian noise applied to inherited weights
4. **Structural Variation**: Random gating patterns applied each generation
5. **Survival**: Pure replacement - no individuals survive between generations

## Installation & Usage

```bash
# Clone repository
git clone <repository-url>
cd life-sim-python

# Install dependencies
pip install -r requirements.txt

# Run simulation
python main.py
```

## Project Structure

```
├── config.py                 # Simulation parameters
├── main.py                   # Main execution loop
├── src/
│   ├── model/
│   │   ├── main.py          # Neural network model and mutations
│   │   └── propagation.py   # Batch inference engine
│   ├── simulation/
│   │   ├── main.py          # Population management and selection
│   │   ├── individual.py    # Individual agent behavior
│   │   ├── heal_zones.py    # Beneficial environment zones
│   │   └── rad_zones.py     # Harmful moving zones
│   └── services/
│       └── individuals.py   # Model persistence
└── requirements.txt
```

## Future Enhancements

- Gate inheritance across generations for structural evolution
  - This will require a mechanism for opening gates; they're currently created randomly.
- Visualization of evolutionary progress and behaviors
