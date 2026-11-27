import pytest

# Adjust the import path to match your project structure
from mcts.minmax import MinMaxStats


def test_init_valid_bounds():
    """Verify initial min and max bounds are stored accurately."""
    stats = MinMaxStats(min_value=-1.0, max_value=1.0)

    assert stats.min == -1.0
    assert stats.max == 1.0


@pytest.mark.parametrize("bound", [0.0, 1.5, -5])
def test_init_equal_bounds_raises(bound):
    """Initializing with identical min and max values must raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        MinMaxStats(min_value=bound, max_value=bound)

    assert str(exc_info.value) == "min max cannot have the same value"


def test_update_expands_min():
    """Updating with a value below current min updates min and keeps max unchanged."""
    stats = MinMaxStats(-1.0, 1.0)
    stats.update(-2.5)

    assert stats.min == -2.5
    assert stats.max == 1.0


def test_update_expands_max():
    """Updating with a value above current max updates max and keeps min unchanged."""
    stats = MinMaxStats(-1.0, 1.0)
    stats.update(3.0)

    assert stats.min == -1.0
    assert stats.max == 3.0


@pytest.mark.parametrize("value", [-1.0, 0.0, 0.5, 1.0])
def test_update_within_or_on_bounds_does_not_change_stats(value):
    """Values inside or equal to [min, max] must not alter the bounds."""
    stats = MinMaxStats(-1.0, 1.0)
    stats.update(value)

    assert stats.min == -1.0
    assert stats.max == 1.0


def test_update_multiple_values():
    """Sequential updates track the global minimum and maximum observed."""
    stats = MinMaxStats(0.0, 1.0)
    for val in (0.5, 2.0, -3.0, 1.5, 4.5, -1.0):
        stats.update(val)

    assert stats.min == -3.0
    assert stats.max == 4.5


@pytest.mark.parametrize(
    "min_val, max_val, value, expected",
    [
        (-1.0, 1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0, 0.5),
        (-1.0, 1.0, 1.0, 1.0),
        (-1.0, 1.0, 0.5, 0.75),
        (0.0, 10.0, 2.5, 0.25),
        (-10.0, -2.0, -6.0, 0.5),
    ],
)
def test_normalize_within_bounds(min_val, max_val, value, expected):
    """Values inside [min, max] map linearly to [0.0, 1.0]."""
    stats = MinMaxStats(min_val, max_val)

    assert stats.normalize(value) == pytest.approx(expected)


def test_normalize_outside_bounds_does_not_mutate_stats():
    """Normalizing out-of-range values extrapolates without changing stored bounds."""
    stats = MinMaxStats(0.0, 2.0)

    assert stats.normalize(-1.0) == pytest.approx(-0.5)
    assert stats.normalize(3.0) == pytest.approx(1.5)
    assert stats.min == 0.0
    assert stats.max == 2.0


def test_normalize_reflects_updated_bounds():
    """Normalization dynamically uses the expanded range after update()."""
    stats = MinMaxStats(0.0, 1.0)
    stats.update(-1.0)
    stats.update(3.0)

    assert stats.normalize(-1.0) == pytest.approx(0.0)
    assert stats.normalize(1.0) == pytest.approx(0.5)
    assert stats.normalize(3.0) == pytest.approx(1.0)