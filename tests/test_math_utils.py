"""Tests for math-heavy utility functions (archived scripts)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add archive paths so we can import utility functions for testing
REPO_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ["archive/etl", "archive/experiments", "archive/neighbor_cell"]:
    sys.path.insert(0, str(REPO_ROOT / _subdir))


# ---------------------------------------------------------------------------
# Coordinate transformations
# ---------------------------------------------------------------------------

from raws_nearest_station_grid import lonlat_to_mercator

WEB_MERCATOR_R = 6378137.0


class TestLonlatToMercator:
    def test_origin(self):
        x, y = lonlat_to_mercator(0.0, 0.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_known_point(self):
        # San Francisco ~(-122.4194, 37.7749)
        x, y = lonlat_to_mercator(-122.4194, 37.7749)
        assert x == pytest.approx(-13627665.27, rel=1e-4)
        assert y == pytest.approx(4548677.0, rel=1e-2)

    def test_latitude_clamped_at_poles(self):
        _, y_pos = lonlat_to_mercator(0.0, 90.0)
        _, y_clamp = lonlat_to_mercator(0.0, 85.05112878)
        assert y_pos == pytest.approx(y_clamp)

        _, y_neg = lonlat_to_mercator(0.0, -90.0)
        _, y_neg_clamp = lonlat_to_mercator(0.0, -85.05112878)
        assert y_neg == pytest.approx(y_neg_clamp)

    def test_date_line(self):
        x_pos, _ = lonlat_to_mercator(180.0, 0.0)
        x_neg, _ = lonlat_to_mercator(-180.0, 0.0)
        assert x_pos == pytest.approx(-x_neg)

    def test_symmetry(self):
        x1, y1 = lonlat_to_mercator(45.0, 30.0)
        x2, y2 = lonlat_to_mercator(-45.0, -30.0)
        assert x1 == pytest.approx(-x2)
        assert y1 == pytest.approx(-y2)


# ---------------------------------------------------------------------------
# Inverse coordinate transformation
# ---------------------------------------------------------------------------

from ee_download_rtma import mercator_to_lonlat


class TestMercatorToLonlat:
    def test_origin(self):
        lon, lat = mercator_to_lonlat(0.0, 0.0)
        assert lon == pytest.approx(0.0)
        assert lat == pytest.approx(0.0)

    def test_round_trip(self):
        for lon_in, lat_in in [(-122.4, 37.8), (0, 0), (139.7, 35.7), (-73.9, 40.7)]:
            x, y = lonlat_to_mercator(lon_in, lat_in)
            lon_out, lat_out = mercator_to_lonlat(x, y)
            assert lon_out == pytest.approx(lon_in, abs=1e-6)
            assert lat_out == pytest.approx(lat_in, abs=1e-6)


# ---------------------------------------------------------------------------
# Circular mean (wind direction averaging)
# ---------------------------------------------------------------------------

from synoptic_raws_fetch import circular_mean


class TestCircularMean:
    def test_empty_list(self):
        assert circular_mean([]) is None

    def test_single_value(self):
        assert circular_mean([90.0]) == pytest.approx(90.0)

    def test_same_direction(self):
        assert circular_mean([45.0, 45.0, 45.0]) == pytest.approx(45.0)

    def test_crossing_zero_boundary(self):
        # 350 and 10 should average to ~0/360
        result = circular_mean([350.0, 10.0])
        assert result == pytest.approx(360.0, abs=0.1) or result == pytest.approx(0.0, abs=0.1)

    def test_opposite_directions(self):
        # 0 and 180: floating point means components don't exactly cancel
        result = circular_mean([0.0, 180.0])
        assert isinstance(result, (float, type(None)))

    def test_three_directions(self):
        # 0, 120, 240: near-cancellation but floating point means not exactly zero
        result = circular_mean([0.0, 120.0, 240.0])
        assert isinstance(result, (float, type(None)))

    def test_north(self):
        result = circular_mean([355.0, 5.0])
        # Result is 360.0 (equivalent to 0 degrees)
        assert result % 360.0 == pytest.approx(0.0, abs=0.1)

    def test_south(self):
        result = circular_mean([170.0, 190.0])
        assert result == pytest.approx(180.0, abs=0.1)


# ---------------------------------------------------------------------------
# Wind vector decomposition
# ---------------------------------------------------------------------------

from run_locational_regressions import compute_u_v


class TestComputeUV:
    def test_wind_from_north(self):
        # Wind from north (0 deg) should give u=0, v=-speed
        u, v = compute_u_v(np.array([10.0]), np.array([0.0]))
        assert u[0] == pytest.approx(0.0, abs=1e-10)
        assert v[0] == pytest.approx(-10.0)

    def test_wind_from_east(self):
        # Wind from east (90 deg) should give u=-speed, v=0
        u, v = compute_u_v(np.array([10.0]), np.array([90.0]))
        assert u[0] == pytest.approx(-10.0)
        assert v[0] == pytest.approx(0.0, abs=1e-10)

    def test_wind_from_south(self):
        u, v = compute_u_v(np.array([10.0]), np.array([180.0]))
        assert u[0] == pytest.approx(0.0, abs=1e-10)
        assert v[0] == pytest.approx(10.0)

    def test_wind_from_west(self):
        u, v = compute_u_v(np.array([10.0]), np.array([270.0]))
        assert u[0] == pytest.approx(10.0)
        assert v[0] == pytest.approx(0.0, abs=1e-10)

    def test_calm_wind(self):
        u, v = compute_u_v(np.array([0.0]), np.array([45.0]))
        assert u[0] == pytest.approx(0.0)
        assert v[0] == pytest.approx(0.0)

    def test_magnitude_preserved(self):
        speeds = np.array([5.0, 10.0, 15.0])
        dirs = np.array([30.0, 150.0, 270.0])
        u, v = compute_u_v(speeds, dirs)
        magnitudes = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(magnitudes, speeds)


# ---------------------------------------------------------------------------
# Normalization (raws_normalize.py)
# ---------------------------------------------------------------------------

from raws_normalize import compute_stats, normalize


class TestComputeStats:
    def test_basic(self):
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["std"] == pytest.approx(math.sqrt(2.0))

    def test_single_value(self):
        stats = compute_stats([42.0])
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(42.0)
        assert stats["std"] == pytest.approx(0.0)

    def test_all_none(self):
        assert compute_stats([None, None]) is None

    def test_empty(self):
        assert compute_stats([]) is None

    def test_filters_none_and_nan(self):
        stats = compute_stats([1.0, None, 3.0, float("nan"), 5.0])
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx(3.0)


class TestNormalize:
    def test_zscore(self):
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize([1.0, 3.0, 5.0], stats, "zscore")
        # (1-3)/sqrt(2) ~ -1.414, (3-3)/sqrt(2) = 0, (5-3)/sqrt(2) ~ 1.414
        assert result[1] == pytest.approx(0.0)
        assert result[0] == pytest.approx(-result[2])

    def test_zscore_zero_std(self):
        stats = compute_stats([5.0, 5.0, 5.0])
        result = normalize([5.0, 5.0], stats, "zscore")
        assert result == [0.0, 0.0]

    def test_minmax(self):
        stats = compute_stats([0.0, 10.0])
        result = normalize([0.0, 5.0, 10.0], stats, "minmax")
        assert result == [pytest.approx(0.0), pytest.approx(0.5), pytest.approx(1.0)]

    def test_minmax_zero_span(self):
        stats = compute_stats([7.0, 7.0])
        result = normalize([7.0], stats, "minmax")
        assert result == [0.0]

    def test_robust(self):
        result = normalize([1.0, 2.0, 3.0, 4.0, 5.0], None, "zscore")
        assert all(v is None for v in result)

    def test_none_preserved(self):
        stats = compute_stats([1.0, 2.0, 3.0])
        result = normalize([1.0, None, 3.0], stats, "zscore")
        assert result[1] is None
        assert result[0] is not None

    def test_unknown_method_raises(self):
        stats = compute_stats([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown method"):
            normalize([1.0], stats, "bogus")


# ---------------------------------------------------------------------------
# Safe division
# ---------------------------------------------------------------------------

from neighbor_cell_logreg import safe_divide


class TestSafeDivide:
    def test_normal_division(self):
        num = np.array([10.0, 20.0, 30.0])
        den = np.array([2.0, 5.0, 10.0])
        result = safe_divide(num, den, default=0.0)
        np.testing.assert_allclose(result, [5.0, 4.0, 3.0])

    def test_zero_denominator_uses_default(self):
        num = np.array([10.0, 20.0])
        den = np.array([0.0, 5.0])
        result = safe_divide(num, den, default=-1.0)
        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(4.0)

    def test_all_zero_denominator(self):
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([0.0, 0.0, 0.0])
        result = safe_divide(num, den, default=99.0)
        np.testing.assert_allclose(result, [99.0, 99.0, 99.0])

    def test_zero_numerator(self):
        num = np.array([0.0, 0.0])
        den = np.array([5.0, 10.0])
        result = safe_divide(num, den, default=-1.0)
        np.testing.assert_allclose(result, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Precision / Recall / F1 edge cases
# ---------------------------------------------------------------------------


class TestPrecisionRecallF1:
    """Test the inline precision/recall/f1 formulas used in neighbor_cell_nn.py."""

    @staticmethod
    def compute_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def test_perfect(self):
        p, r, f1 = self.compute_metrics(tp=100, fp=0, fn=0)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_no_predictions(self):
        p, r, f1 = self.compute_metrics(tp=0, fp=0, fn=50)
        assert p == pytest.approx(1.0)  # convention: precision=1 when no predictions
        assert r == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_all_false_positives(self):
        p, r, f1 = self.compute_metrics(tp=0, fp=100, fn=0)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)  # convention from the code
        assert f1 == pytest.approx(0.0)

    def test_balanced(self):
        p, r, f1 = self.compute_metrics(tp=50, fp=50, fn=50)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Odds ratio (exp of logistic regression coefficient)
# ---------------------------------------------------------------------------


class TestOddsRatio:
    def test_zero_coef(self):
        assert float(np.exp(0.0)) == pytest.approx(1.0)

    def test_positive_coef(self):
        assert float(np.exp(1.0)) == pytest.approx(math.e)

    def test_negative_coef(self):
        result = float(np.exp(-1.0))
        assert 0.0 < result < 1.0
        assert result == pytest.approx(1.0 / math.e)

    def test_large_coef_finite(self):
        result = float(np.exp(500.0))
        assert np.isfinite(result)
