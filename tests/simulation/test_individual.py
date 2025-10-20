from core.config import MAX_LENGTH
from simulation.individual import Individual, IndividualUpdateContext


def test_update_heal_zone():
    indiv = Individual(model=None)

    heal_zone = type('HealZone', (), {'position': (5, 5), 'radius': 3})
    rad_zone = type('RadZone', (), {'position': (15, 15), 'radius': 3, 'direction': (1, 0)})

    heal_zones = [heal_zone]
    rad_zones = [rad_zone]

    initial_times_healed = indiv.times_healed

    indiv.position = (5, 5)
    context = indiv.update(heal_zones, rad_zones)
    assert context.times_healed == initial_times_healed + 1
    assert context.heal_zone_dist == 0.0
    assert context.heal_zone_dir == (0.0, 0.0)

def test_update_rad_zone():
    indiv = Individual(model=None)

    heal_zone = type('HealZone', (), {'position': (5, 5), 'radius': 3})
    rad_zone = type('RadZone', (), {'position': (15, 15), 'radius': 3, 'direction': (1, 0)})

    heal_zones = [heal_zone]
    rad_zones = [rad_zone]

    initial_times_healed = indiv.times_healed

    indiv.position = (15, 15)
    context = indiv.update(heal_zones, rad_zones)
    assert context.times_healed == initial_times_healed - 2
    assert context.rad_zone_dist == 0.0
    assert context.rad_zone_dir == (0.0, 0.0)
    assert context.rad_zone_move_dir == (1.0, 0.0)

def test_update_no_zones():
    indiv = Individual(model=None)

    heal_zone_radius = 5
    rad_zone_radius = 5

    heal_zone = type('HealZone', (), {'position': (50, 50), 'radius': heal_zone_radius})
    rad_zone = type('RadZone', (), {'position': (60, 60), 'radius': rad_zone_radius, 'direction': (1, 0)})

    heal_zones = [heal_zone]
    rad_zones = [rad_zone]

    initial_times_healed = indiv.times_healed

    indiv.position = (0, 0)
    context = indiv.update(heal_zones, rad_zones)

    print(context)
    assert context.times_healed == initial_times_healed
    assert context.heal_zone_dist * MAX_LENGTH > heal_zone_radius
    assert context.rad_zone_dist * MAX_LENGTH > rad_zone_radius


def test_calculate_input_values_previous_position():
    indiv = Individual(model=None)

    indiv.position = (5, 5)
    indiv.previous_position = (3, 4)

    context = IndividualUpdateContext(
        heal_zone_dir=(1, 0),
        heal_zone_dist=1.0,
        rad_zone_dir=(0, -1),
        rad_zone_dist=2.0,
        rad_zone_move_dir=(0, 1),
        next_position=(6, 5),
        times_healed=3
    )

    expected = (2, 1)
    actual = indiv.calculate_input_values(context)

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]

def test_calculate_input_values_heal_zone():
    indiv = Individual(model=None)

    indiv.position = (5, 5)
    indiv.previous_position = (5, 5)

    heal_zone_dist = 10.0
    heal_zone_dir = (0.6, 0.8)

    context = IndividualUpdateContext(
        heal_zone_dir=heal_zone_dir,
        heal_zone_dist=heal_zone_dist,
        rad_zone_dir=(0, -1),
        rad_zone_dist=2.0,
        rad_zone_move_dir=(0, 1),
        next_position=(6, 5),
        times_healed=3
    )

    actual = indiv.calculate_input_values(context)

    assert actual[2] == heal_zone_dist
    assert actual[3] == heal_zone_dir[0]
    assert actual[4] == heal_zone_dir[1]

def test_calculate_input_values_rad_zone():
    indiv = Individual(model=None)

    indiv.position = (5, 5)
    indiv.previous_position = (5, 5)

    rad_zone_dist = 15.0
    rad_zone_dir = (-0.6, -0.8)
    rad_zone_move_dir = (0.0, 1.0)

    context = IndividualUpdateContext(
        heal_zone_dir=(1, 0),
        heal_zone_dist=1.0,
        rad_zone_dir=rad_zone_dir,
        rad_zone_dist=rad_zone_dist,
        rad_zone_move_dir=rad_zone_move_dir,
        next_position=(6, 5),
        times_healed=3
    )

    actual = indiv.calculate_input_values(context)

    assert actual[5] == rad_zone_dist
    assert actual[6] == rad_zone_dir[0]
    assert actual[7] == rad_zone_dir[1]
    assert actual[8] == rad_zone_move_dir[0]
    assert actual[9] == rad_zone_move_dir[1]