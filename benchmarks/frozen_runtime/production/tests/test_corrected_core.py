"""Contract tests for the corrected Day0-locked benchmark core.

Run with the locked project environment, without adding test dependencies:

    /path/to/ChronoClone/.runtime/chronoclone/bin/python \
      -m unittest -v tests/test_corrected_core.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import corrected_core as core  # noqa: E402


def example_counts() -> np.ndarray:
    """Five clones spanning valid, contaminated, gapped, and low-count cases."""

    counts = np.zeros((5, len(core.TIMES), len(core.STATES)), dtype=np.int64)
    hsc = core.STATES.index("HSC")
    hspc = core.STATES.index("HSPC")
    mono = core.STATES.index("Mono_prog")

    counts[0, 0, hsc] = 10
    counts[0, 1, hspc] = 5
    counts[0, 2, mono] = 2

    counts[1, 0, hspc] = 10
    counts[1, 1, hsc] = 4

    counts[2, 0, hsc] = 9
    counts[2, 0, mono] = 1  # forbidden Day0 contamination
    counts[2, 1, hspc] = 4

    counts[3, 0, hsc] = 10
    counts[3, 2, mono] = 2  # internal Day4 gap

    counts[4, 0, hsc] = 2  # total-count and persistence failure
    return counts


class CorrectedCoreTests(unittest.TestCase):
    def test_lineage_mask_matches_start_purity_continuity_and_count_contract(self):
        observed = core.lineage_eligibility_mask(example_counts())
        np.testing.assert_array_equal(observed, [True, True, False, False, False])

    def test_gaussian_noise_never_changes_any_day0_entry(self):
        clean = example_counts()
        noisy, seed = core.gaussian_stage_day0_locked(
            clean, gaussian_pct=100, repeat=7, master_seed=20260729
        )
        core.assert_day0_frozen(clean, noisy)
        np.testing.assert_array_equal(noisy[:, 0, :], clean[:, 0, :])
        self.assertIsInstance(seed, int)
        self.assertTrue(np.any(noisy[:, 1:, :] != clean[:, 1:, :]))

    def test_gaussian_does_not_call_legacy_full_tensor_stage(self):
        with patch.object(
            core.LEGACY_CORE,
            "gaussian_stage",
            side_effect=AssertionError("legacy full-tensor Gaussian was called"),
        ):
            noisy, _ = core.gaussian_stage_day0_locked(
                example_counts(),
                gaussian_pct=60,
                repeat=2,
                master_seed=20260729,
            )
        np.testing.assert_array_equal(noisy[:, 0, :], example_counts()[:, 0, :])

    def test_fractional_clean_means_lock_to_zero_noise_integer_day0(self):
        clean_means = example_counts().astype(float)
        clean_means[0, 0, core.STATES.index("HSC")] = 6.666666666666667
        clean_means[0, 0, core.STATES.index("HSPC")] = 3.333333333333333
        expected = core.rounded_clean_counts(clean_means)
        observed, _ = core.gaussian_stage_day0_locked(
            clean_means, gaussian_pct=100, repeat=7, master_seed=20260729
        )
        np.testing.assert_array_equal(observed[:, 0, :], expected[:, 0, :])

    def test_uniform_noise_freezes_day0_totals_and_gaussian_eligibility(self):
        clean = example_counts()
        gaussian, _ = core.gaussian_stage_day0_locked(
            clean, gaussian_pct=80, repeat=4, master_seed=20260729
        )
        before = core.lineage_eligibility_mask(gaussian)
        for uniform_pct in [0, 2, 6, 50, 100]:
            with self.subTest(uniform_pct=uniform_pct):
                redistributed, seed = core.uniform_redistribution_stage_day0_locked(
                    gaussian,
                    gaussian_pct=80,
                    uniform_pct=uniform_pct,
                    repeat=4,
                    master_seed=20260729,
                )
                after = core.lineage_eligibility_mask(redistributed)
                np.testing.assert_array_equal(redistributed[:, 0, :], clean[:, 0, :])
                np.testing.assert_array_equal(
                    redistributed.sum(axis=2), gaussian.sum(axis=2)
                )
                np.testing.assert_array_equal(after, before)
                self.assertIsInstance(seed, int)

    def test_uniform_does_not_call_legacy_full_tensor_stage(self):
        gaussian, _ = core.gaussian_stage_day0_locked(
            example_counts(), gaussian_pct=60, repeat=2, master_seed=20260729
        )
        with patch.object(
            core.LEGACY_CORE,
            "uniform_redistribution_stage",
            side_effect=AssertionError("legacy full-tensor uniform was called"),
        ):
            redistributed, _ = core.uniform_redistribution_stage_day0_locked(
                gaussian,
                gaussian_pct=60,
                uniform_pct=60,
                repeat=2,
                master_seed=20260729,
            )
        np.testing.assert_array_equal(redistributed[:, 0, :], gaussian[:, 0, :])

    def test_primary_current_and_secondary_clean_fixed_masks_are_distinct(self):
        clean = example_counts()
        current = clean.copy()
        current[0, 1, :] = 0  # Gaussian clipping can erase a later block.
        eligibility = core.condition_eligibility_masks(clean, current)
        np.testing.assert_array_equal(
            eligibility.clean_fixed_mask, [True, True, False, False, False]
        )
        np.testing.assert_array_equal(
            eligibility.current_lineage_mask, [False, True, False, False, False]
        )
        self.assertEqual(eligibility.n_current, 1)
        self.assertEqual(eligibility.n_clean_fixed, 2)
        self.assertEqual(eligibility.n_mask_disagreements, 1)
        core.assert_fixed_cohort_mask(eligibility.clean_fixed_mask, clean)

    def test_condition_contract_rejects_any_changed_day0_value(self):
        clean = example_counts()
        current = clean.copy()
        current[0, 0, core.STATES.index("HSC")] -= 1
        with self.assertRaisesRegex(AssertionError, "Day0 changed"):
            core.condition_eligibility_masks(clean, current)

    def test_fixed_cohort_assertion_rejects_current_mask_substitution(self):
        clean = example_counts()
        wrong = core.clean_fixed_eligibility_mask(clean).copy()
        wrong[0] = False
        with self.assertRaisesRegex(AssertionError, "differs from clean eligibility"):
            core.assert_fixed_cohort_mask(wrong, clean)

    def test_silhouette_postfilter_is_inclusive_unrounded_and_does_not_refit(self):
        features = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
        labels = np.asarray([7, 7, 12], dtype=np.int64)
        supplied = np.asarray([-0.1, -0.100000000001, 0.4], dtype=float)

        def fake_silhouette(_features, _labels, *, metric):
            self.assertEqual(metric, "euclidean")
            return supplied.copy()

        with patch.object(core, "silhouette_samples", fake_silhouette):
            result = core.chronoclone_silhouette_postfilter(features, labels)
        np.testing.assert_array_equal(result.labels, labels)
        np.testing.assert_array_equal(result.sample_silhouette, supplied)
        np.testing.assert_array_equal(result.retained_mask, [True, False, True])
        self.assertEqual(result.cutoff, -0.1)

    def test_selective_metrics_report_conditional_accuracy_and_both_coverages(self):
        features = np.asarray([[0.0], [10.0], [9.0], [11.0]], dtype=float)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        truth = np.asarray(["A", "B", "B", "B"])
        selection, metrics = core.chronoclone_postfilter_metrics(
            features, labels, truth, n_input_clones=5
        )
        np.testing.assert_array_equal(
            selection.retained_mask, [True, False, True, True]
        )
        self.assertEqual(metrics["n_input_clones"], 5)
        self.assertEqual(metrics["n_fitted_clones"], 4)
        self.assertEqual(metrics["n_retained_clones"], 3)
        self.assertAlmostEqual(metrics["fitted_fraction_of_input"], 0.8)
        self.assertAlmostEqual(metrics["retained_fraction_of_fitted"], 0.75)
        self.assertAlmostEqual(metrics["retained_fraction_of_input"], 0.6)
        self.assertEqual(metrics["silhouette_cutoff"], -0.1)
        for name in (
            "selective_ari",
            "selective_nmi",
            "selective_hungarian_macro_f1",
        ):
            self.assertTrue(np.isfinite(metrics[name]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
