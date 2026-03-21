"""Tests for data quality fixes: RTMA imputation, wind direction, discounted rain, etc."""

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from neighbor_cell_logreg import (
    RTMA_VALID_RANGES,
    _spatial_fill_nan,
    build_discounted_rain_state,
    sanitize_rtma_variable,
    to_binary_target,
)


# ---------------------------------------------------------------------------
# Spatial NaN fill (imputation helper)
# ---------------------------------------------------------------------------


class TestSpatialFillNan:
    """Tests for the _spatial_fill_nan helper that fills NaN from neighbors."""

    def test_single_nan_surrounded_by_valid(self):
        """A single NaN pixel surrounded by valid values gets the mean of its neighbors."""
        arr = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float32)
        result = _spatial_fill_nan(arr)
        # Mean of all 8 neighbors: (1+2+3+4+6+7+8+9)/8 = 5.0
        assert result[1, 1] == pytest.approx(5.0)

    def test_corner_nan(self):
        """A NaN in the corner has only 3 neighbors."""
        arr = np.array([
            [np.nan, 2.0],
            [4.0, 6.0],
        ], dtype=np.float32)
        result = _spatial_fill_nan(arr)
        # Mean of 3 valid neighbors: (2+4+6)/3 = 4.0
        assert result[0, 0] == pytest.approx(4.0)

    def test_edge_nan(self):
        """A NaN on the edge has 5 neighbors."""
        arr = np.array([
            [1.0, np.nan, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float32)
        result = _spatial_fill_nan(arr)
        # Mean of 5 valid neighbors: (1+3+4+5+6)/5 = 3.8
        assert result[0, 1] == pytest.approx(3.8)

    def test_no_nan_unchanged(self):
        """Array with no NaN should be returned as-is."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = _spatial_fill_nan(arr)
        np.testing.assert_array_equal(result, arr)

    def test_all_nan_stays_nan(self):
        """If the entire grid is NaN, nothing can be filled."""
        arr = np.full((3, 3), np.nan, dtype=np.float32)
        result = _spatial_fill_nan(arr)
        assert np.isnan(result).all()

    def test_multi_pass_propagation(self):
        """Multiple NaN pixels in a line get filled by iterative passes."""
        arr = np.array([
            [10.0, np.nan, np.nan, np.nan, 10.0],
        ], dtype=np.float32)
        result = _spatial_fill_nan(arr, max_passes=5)
        # After passes, NaN should fill inward from both sides
        assert np.isfinite(result).all()

    def test_1d_array_passthrough(self):
        """1D arrays (non-spatial) should pass through unchanged."""
        arr = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        result = _spatial_fill_nan(arr)
        assert np.isnan(result[1])  # Not filled (1D has no 2D neighbors)


# ---------------------------------------------------------------------------
# RTMA sentinel value sanitization (imputation)
# ---------------------------------------------------------------------------


class TestSanitizeRtmaVariable:
    """Tests for sanitize_rtma_variable which replaces bad values and imputes."""

    def test_acpc01_sentinel_9999_imputed(self):
        """9999.0 sentinel in ACPC01 should be imputed from neighbors, not clamped."""
        arr = np.array([
            [0.0, 5.0, 10.0],
            [2.0, 9999.0, 8.0],
            [1.0, 3.0, 7.0],
        ], dtype=np.float32)
        result = sanitize_rtma_variable(arr, "ACPC01")
        # Center pixel should be mean of 8 neighbors: (0+5+10+2+8+1+3+7)/8 = 4.5
        assert result[1, 1] == pytest.approx(4.5)
        assert result[0, 0] == pytest.approx(0.0)  # Valid values unchanged

    def test_acpc01_negative_imputed(self):
        """Negative precipitation should be imputed from neighbors."""
        arr = np.array([
            [0.0, 10.0],
            [-5.0, 8.0],
        ], dtype=np.float32)
        result = sanitize_rtma_variable(arr, "ACPC01")
        # -5.0 is out of range [0,100], imputed from 3 neighbors: (0+10+8)/3 = 6.0
        assert result[1, 0] == pytest.approx(6.0)

    def test_acpc01_nan_imputed(self):
        """NaN values should be imputed from valid neighbors."""
        arr = np.array([
            [5.0, float("nan")],
            [3.0, 7.0],
        ], dtype=np.float32)
        result = sanitize_rtma_variable(arr, "ACPC01")
        # NaN at [0,1] imputed from neighbors: (5+3+7)/3 = 5.0
        assert result[0, 1] == pytest.approx(5.0)

    def test_acpc01_valid_unchanged(self):
        """Valid ACPC01 values within range should be unchanged."""
        arr = np.array([[0.0, 10.5], [50.0, 100.0]], dtype=np.float32)
        result = sanitize_rtma_variable(arr, "ACPC01")
        np.testing.assert_allclose(result, arr)

    def test_tmp_out_of_range_imputed(self):
        """Out-of-range TMP values should be imputed, not clamped."""
        arr = np.array([
            [20.0, 25.0, 22.0],
            [18.0, 200.0, 21.0],  # 200°C is a sentinel
            [19.0, 23.0, 24.0],
        ], dtype=np.float32)
        result = sanitize_rtma_variable(arr, "TMP")
        # Center imputed: mean of 8 neighbors ≈ 21.5
        expected_center = (20 + 25 + 22 + 18 + 21 + 19 + 23 + 24) / 8
        assert result[1, 1] == pytest.approx(expected_center)

    def test_unknown_variable_passthrough(self):
        """Unknown variables should pass through unchanged."""
        arr = np.array([[9999.0, -9999.0]], dtype=np.float32)
        result = sanitize_rtma_variable(arr, "UNKNOWN_VAR")
        np.testing.assert_array_equal(result, arr)

    def test_no_copy_when_clean(self):
        """When no sanitization needed, should return original array."""
        arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        result = sanitize_rtma_variable(arr, "TMP")
        assert result is arr

    def test_isolated_sentinel_in_large_grid(self):
        """A single sentinel in a large valid grid should be fully imputed."""
        arr = np.full((10, 10), 5.0, dtype=np.float32)
        arr[5, 5] = 9999.0  # Single sentinel
        result = sanitize_rtma_variable(arr, "ACPC01")
        assert np.isfinite(result).all()
        assert result[5, 5] == pytest.approx(5.0)  # Mean of neighbors (all 5.0)

    def test_valid_ranges_all_variables_defined(self):
        """All required RTMA variables should have valid ranges defined."""
        required = ["TMP", "WIND", "WDIR", "SPFH", "ACPC01"]
        for var in required:
            assert var in RTMA_VALID_RANGES, f"Missing valid range for {var}"
            lo, hi = RTMA_VALID_RANGES[var]
            assert lo < hi, f"Invalid range for {var}: [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# Discounted rain state (negative clamping, edge cases)
# ---------------------------------------------------------------------------


class TestBuildDiscountedRainState:
    """Tests for the discounted rain accumulation function."""

    def test_basic_accumulation(self):
        """Rain should accumulate with exponential decay."""
        state = np.array([0.0, 0.0], dtype=np.float32)
        precip = np.array([10.0, 5.0], dtype=np.float32)
        history = deque()
        decay = 0.99
        expiry = 0.5 ** 31  # negligible

        result = build_discounted_rain_state(
            state, history, precip,
            decay_per_hour=decay, expiry_factor=expiry, lookback_hours=720,
        )
        expected = decay * state + decay * precip
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_no_negative_state(self):
        """Accumulated rain must never go negative after expiry subtraction."""
        state = np.array([1.0], dtype=np.float32)
        # Large expired precip that would make state negative without clamping
        expired = np.array([1000.0], dtype=np.float32)
        history = deque([expired] + [np.zeros(1, dtype=np.float32)] * 719)
        current = np.array([0.0], dtype=np.float32)

        result = build_discounted_rain_state(
            state, history, current,
            decay_per_hour=0.99, expiry_factor=0.5, lookback_hours=720,
        )
        assert result[0] >= 0.0, f"Discounted rain went negative: {result[0]}"

    def test_zero_precip_decays(self):
        """With zero new precip, state should decay toward zero."""
        state = np.array([100.0], dtype=np.float32)
        history = deque()
        zero = np.array([0.0], dtype=np.float32)

        result = build_discounted_rain_state(
            state, history, zero,
            decay_per_hour=0.99, expiry_factor=0.01, lookback_hours=720,
        )
        assert result[0] < 100.0
        assert result[0] >= 0.0

    def test_history_deque_grows(self):
        """Precipitation history should grow until lookback_hours."""
        history = deque()
        state = np.array([0.0], dtype=np.float32)
        for _ in range(10):
            precip = np.array([1.0], dtype=np.float32)
            state = build_discounted_rain_state(
                state, history, precip,
                decay_per_hour=0.99, expiry_factor=0.01, lookback_hours=720,
            )
        assert len(history) == 10

    def test_history_pops_at_lookback(self):
        """When history reaches lookback_hours, oldest entry should be removed."""
        lookback = 5
        history = deque([np.array([1.0], dtype=np.float32)] * lookback)
        state = np.array([50.0], dtype=np.float32)
        precip = np.array([2.0], dtype=np.float32)

        result = build_discounted_rain_state(
            state, history, precip,
            decay_per_hour=0.99, expiry_factor=0.01, lookback_hours=lookback,
        )
        assert len(history) == lookback
        assert result[0] >= 0.0


# ---------------------------------------------------------------------------
# Binary target threshold
# ---------------------------------------------------------------------------


class TestToBinaryTarget:
    def test_basic_threshold(self):
        y = np.array([0.0, 0.05, 0.1, 0.5, 1.0])
        result = to_binary_target(y, 0.1)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 1])

    def test_all_below(self):
        y = np.array([0.0, 0.01, 0.09])
        result = to_binary_target(y, 0.1)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_all_above(self):
        y = np.array([0.5, 0.8, 1.0])
        result = to_binary_target(y, 0.1)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_exact_threshold_is_positive(self):
        y = np.array([0.1])
        result = to_binary_target(y, 0.1)
        assert result[0] == 1


# ---------------------------------------------------------------------------
# RAWS normalization (updated with Bessel's correction)
# ---------------------------------------------------------------------------

from raws_normalize import compute_stats, normalize


class TestComputeStatsBessel:
    """Verify Bessel's correction is applied correctly."""

    def test_two_values(self):
        """With N=2, sample variance divides by 1 (not 2)."""
        stats = compute_stats([0.0, 2.0])
        # Population var = 1.0, sample var = 2.0
        assert stats["std"] == pytest.approx(math.sqrt(2.0))

    def test_single_value_zero_std(self):
        """Single value should give std=0 (0/max(0,1)=0)."""
        stats = compute_stats([42.0])
        assert stats["std"] == pytest.approx(0.0)

    def test_identical_values(self):
        """Identical values should give std=0 regardless of count."""
        stats = compute_stats([5.0, 5.0, 5.0, 5.0])
        assert stats["std"] == pytest.approx(0.0)


class TestRobustNormalization:
    """Verify the fixed robust normalization method."""

    def test_odd_count_median(self):
        result = normalize([1.0, 2.0, 3.0, 4.0, 5.0], None, "robust")
        # stats=None means all None output
        assert all(v is None for v in result)

    def test_odd_count_with_stats(self):
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize([3.0], stats, "robust")
        # median=3, deviations=[0,1,1,2,2], MAD=1, scaled MAD=1.4826
        assert result[0] == pytest.approx(0.0)

    def test_even_count_mad(self):
        """Even count should properly compute median of absolute deviations."""
        stats = compute_stats([1.0, 2.0, 3.0, 4.0])
        result = normalize([2.5], stats, "robust")
        # median=(2+3)/2=2.5, deviations=[1.5,0.5,0.5,1.5], sorted=[0.5,0.5,1.5,1.5]
        # MAD=(0.5+1.5)/2=1.0
        assert result[0] == pytest.approx(0.0)

    def test_none_preserved_in_robust(self):
        stats = compute_stats([1.0, 2.0, 3.0])
        result = normalize([1.0, None, 3.0], stats, "robust")
        assert result[1] is None
        assert result[0] is not None


# ---------------------------------------------------------------------------
# Validate manifest fixes
# ---------------------------------------------------------------------------

from validate_multires_manifest import ValidationError, validate_manifest


class TestManifestVersionFix:
    """The version comparison should accept string '2.0', int 2, and float 2.0."""

    def _minimal_manifest(self, version):
        return {
            "manifest_version": version,
            "generated_at": "2024-01-01T00:00:00Z",
            "study_area": {"name": "test"},
            "time_window": {"start": "2024-01-01", "end": "2024-01-02"},
            "target_grid": {"resolution_m": 2000},
            "sources": [
                {
                    "id": "s1",
                    "provider": "test",
                    "dataset": "test",
                    "source_type": "dynamic",
                    "native_resolution_m": 2000,
                    "native_temporal_resolution": "1h",
                    "resampling_method": "bilinear",
                    "source_priority": 1,
                    "variables": ["v1"],
                }
            ],
            "variables": [
                {"source_id": "s1", "name": "v1", "dtype": "float32", "units": "K", "kind": "continuous"}
            ],
            "provenance": {},
        }

    def test_string_version(self):
        validate_manifest(self._minimal_manifest("2.0"))

    def test_int_version(self):
        validate_manifest(self._minimal_manifest(2))

    def test_float_version(self):
        validate_manifest(self._minimal_manifest(2.0))

    def test_wrong_version_raises(self):
        with pytest.raises(ValidationError):
            validate_manifest(self._minimal_manifest("3.0"))

    def test_invalid_resampling_method(self):
        m = self._minimal_manifest("2.0")
        m["sources"][0]["resampling_method"] = "invalid_method"
        with pytest.raises(ValidationError, match="resampling_method"):
            validate_manifest(m)

    def test_invalid_dtype(self):
        m = self._minimal_manifest("2.0")
        m["variables"][0]["dtype"] = "complex128"
        with pytest.raises(ValidationError, match="dtype"):
            validate_manifest(m)

    def test_valid_range_wrong_order(self):
        m = self._minimal_manifest("2.0")
        m["variables"][0]["valid_range"] = [100, 0]  # min > max
        with pytest.raises(ValidationError, match="valid_range"):
            validate_manifest(m)


# ---------------------------------------------------------------------------
# HTTP retry logic (synoptic_raws_fetch)
# ---------------------------------------------------------------------------

from synoptic_raws_fetch import http_get_json


class TestHttpGetJsonSignature:
    """Verify the retry function has the expected interface."""

    def test_has_max_retries_param(self):
        """http_get_json should accept max_retries keyword arg."""
        import inspect
        sig = inspect.signature(http_get_json)
        assert "max_retries" in sig.parameters
