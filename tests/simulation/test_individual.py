from simulation.individual import Individual


def test_calculate_input_values_previous_position():
    indiv = Individual(model=None)

    indiv.position = (5, 5)
    indiv.previous_position = (3, 4)

    expected = (2, 1)
    actual = indiv.calculate_input_values()

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]