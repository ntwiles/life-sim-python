import math


def calculate_theoretical_max_fitness() -> float:
    from config import NUM_HEAL_ZONES, HEAL_ZONE_RADIUS, GRID_SIZE, SIMULATOR_STEPS
    
    heal_zone_area = NUM_HEAL_ZONES * math.pi * (HEAL_ZONE_RADIUS ** 2)
    total_grid_area = GRID_SIZE ** 2
    
    heal_zone_coverage = heal_zone_area / total_grid_area

    # Heuristic assuming even distribution of heal zones
    distance_between_centers  = (HEAL_ZONE_RADIUS * math.sqrt(math.pi)) / math.sqrt(heal_zone_coverage)
    distance_between_edges = distance_between_centers - (2 * HEAL_ZONE_RADIUS)

    average_distance_to_heal_zone = distance_between_edges / 2

    perfect_fitness = SIMULATOR_STEPS

    return perfect_fitness - average_distance_to_heal_zone
