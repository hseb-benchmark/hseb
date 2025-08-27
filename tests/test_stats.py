import numpy as np
from hseb.core.stats import Stats, PERCENTILES


def test_from_list_normal_positive_values(self):
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    stats = Stats.from_list(values)

    assert stats.avg == 5.5
    assert len(stats.percentiles) == len(PERCENTILES)


def test_from_list_single_value(self):
    values = [42.5]
    stats = Stats.from_list(values)

    assert stats.avg == 42.5
    assert len(stats.percentiles) == len(PERCENTILES)

    for percentile_value in stats.percentiles.values():
        assert percentile_value == 42.5


def test_from_list_floating_point_precision(self):
    values = [1.123456789, 2.987654321, 3.141592653, 4.999999999]
    stats = Stats.from_list(values)

    assert isinstance(stats.avg, float)
    assert abs(stats.avg - np.mean(values)) < 1e-10

    expected_percentiles = np.percentile(values, PERCENTILES)
    for i, percentile in enumerate(PERCENTILES):
        assert abs(stats.percentiles[str(percentile)] - expected_percentiles[i]) < 1e-10
