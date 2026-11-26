"""Contract and cross-language tests for the corrected Cyclone adapter.

Run with the locked MacBook environment::

    /path/to/ChronoClone/.runtime/chronoclone/bin/python \
      -m unittest -v tests/test_cyclone_adapter.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS = WORKSPACE / "scripts"
for path in (WORKSPACE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import corrected_core  # noqa: E402
import cyclone_adapter  # noqa: E402


R_SCRIPT = (
    Path(cyclone_adapter.legacy_core.__file__).resolve().parents[1]
    / "R"
    / "validate_cyclone_core.R"
)
RSCRIPT = shutil.which("Rscript")


def corrected_fixture(gaussian_pct: int, uniform_pct: int, repeat: int = 0):
    reference = cyclone_adapter.legacy_core.load_reference_programs(
        WORKSPACE / "reference"
    )
    clean = corrected_core.rounded_clean_counts(reference.clean_means)
    gaussian, _ = corrected_core.gaussian_stage_day0_locked(
        reference.clean_means,
        gaussian_pct=gaussian_pct,
        repeat=repeat,
        master_seed=20260729,
    )
    counts, _ = corrected_core.uniform_redistribution_stage_day0_locked(
        gaussian,
        gaussian_pct=gaussian_pct,
        uniform_pct=uniform_pct,
        repeat=repeat,
        master_seed=20260729,
    )
    corrected_core.assert_day0_frozen(clean, counts)
    eligibility = corrected_core.condition_eligibility_masks(clean, counts)
    mask = eligibility.current_lineage_mask
    return counts[mask], reference.clone_ids[mask]


def finite_synthetic_counts() -> tuple[np.ndarray, np.ndarray]:
    """Small abundance fixture with finite source-R distances."""

    abundance = np.asarray(
        [
            [5, 2, 9, 4],
            [1, 8, 3, 7],
            [6, 4, 2, 9],
            [3, 9, 6, 1],
            [8, 5, 1, 6],
            [2, 7, 8, 3],
            [9, 1, 5, 8],
            [4, 6, 7, 2],
        ],
        dtype=np.int64,
    )
    counts = np.zeros((len(abundance), 4, 13), dtype=np.int64)
    counts[:, :, 0] = abundance
    clone_ids = np.asarray([f"synthetic_{index}" for index in range(len(counts))])
    return counts, clone_ids


class CycloneAdapterTests(unittest.TestCase):
    def test_pairwise_complete_correlation_omits_missing_rows_per_pair(self):
        matrix = np.asarray(
            [
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 3.0],
                [1.0, 2.0, 2.0],
                [2.0, 4.0, 1.0],
            ]
        )
        observed = cyclone_adapter.r_pairwise_complete_correlation(matrix)
        expected = np.asarray(
            [
                [1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0)

    def test_pairwise_complete_correlation_returns_nan_for_insufficient_pairs(self):
        matrix = np.asarray(
            [
                [1.0, np.nan, 4.0],
                [np.nan, 2.0, 5.0],
                [3.0, np.nan, np.nan],
            ]
        )
        observed = cyclone_adapter.r_pairwise_complete_correlation(matrix)
        self.assertEqual(observed[0, 0], 1.0)
        self.assertTrue(np.isnan(observed[0, 1]))
        self.assertTrue(np.isnan(observed[0, 2]))
        self.assertTrue(np.isnan(observed[1, 1]))
        self.assertTrue(np.isnan(observed[1, 2]))

    def test_vectorized_mask_groups_match_scalar_pairwise_oracle(self):
        rng = np.random.default_rng(9187)
        matrix = rng.normal(size=(4, 517))
        matrix[rng.random(size=matrix.shape) < 0.18] = np.nan
        matrix[:, 10] = 2.0
        matrix[0, 11] = np.inf
        matrix[1, 12] = -np.inf

        observed = cyclone_adapter.r_pairwise_complete_correlation(matrix)
        expected = np.full((matrix.shape[1], matrix.shape[1]), np.nan)
        with np.errstate(all="ignore"):
            for left in range(matrix.shape[1]):
                for right in range(left, matrix.shape[1]):
                    complete = ~(
                        np.isnan(matrix[:, left])
                        | np.isnan(matrix[:, right])
                    )
                    if int(complete.sum()) < 2:
                        continue
                    x = matrix[complete, left]
                    y = matrix[complete, right]
                    x = x - x.mean()
                    y = y - y.mean()
                    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
                    value = np.dot(x, y) / denominator
                    if np.isfinite(value) and denominator > 0:
                        expected[left, right] = np.clip(value, -1.0, 1.0)
                        expected[right, left] = expected[left, right]
        np.testing.assert_allclose(
            observed, expected, atol=1e-15, rtol=0, equal_nan=True
        )

    def test_clean_deterministic_tensor_is_an_expected_method_failure(self):
        counts, _ = corrected_fixture(0, 0)
        with self.assertRaises(cyclone_adapter.legacy_core.NativeSelectionFailure):
            cyclone_adapter.cyclone_correlation_core(counts, n_clusters=6)

    def test_day0_locked_gaussian_tensor_matches_expected_r_method_failure(self):
        counts, clone_ids = corrected_fixture(20, 0)
        self.assertEqual(len(counts), len(clone_ids))
        with self.assertRaises(cyclone_adapter.legacy_core.NativeSelectionFailure):
            cyclone_adapter.fit_cyclone_method(
                counts,
                clone_ids,
                {"n_clusters": 6, "dynamic_clone_filter": False},
            )

    def test_finite_synthetic_tensor_returns_legacy_methodfit(self):
        counts, clone_ids = finite_synthetic_counts()
        fit = cyclone_adapter.fit_cyclone_method(
            counts,
            clone_ids,
            {"n_clusters": 6, "dynamic_clone_filter": False},
        )
        self.assertIsInstance(fit, cyclone_adapter.legacy_core.MethodFit)
        self.assertEqual(fit.method, "Cyclone abundance-only")
        self.assertEqual(fit.recovered_k, 6)
        self.assertEqual(len(fit.labels), len(counts))
        self.assertTrue(fit.diagnostic["pairwise_complete_correlation"])

    @unittest.skipUnless(RSCRIPT and R_SCRIPT.is_file(), "Rscript unavailable")
    def test_exact_r_agreement_on_corrected_day0_locked_tensors(self):
        from sklearn.metrics import adjusted_rand_score

        fixtures = [(0, 0), (20, 0), (20, 20), (60, 60), (100, 100)]
        with tempfile.TemporaryDirectory(prefix="cyclone_pairwise_validation_") as tmp:
            tmpdir = Path(tmp)
            for gaussian_pct, uniform_pct in fixtures:
                with self.subTest(
                    gaussian_pct=gaussian_pct, uniform_pct=uniform_pct
                ):
                    counts, clone_ids = corrected_fixture(
                        gaussian_pct, uniform_pct
                    )
                    abundance = counts.sum(axis=2, dtype=np.int64)
                    stem = f"g{gaussian_pct:03d}_u{uniform_pct:03d}"
                    input_path = tmpdir / f"{stem}_abundance.csv"
                    labels_path = tmpdir / f"{stem}_labels.csv"
                    correlation_path = tmpdir / f"{stem}_correlation.csv"
                    distance_path = tmpdir / f"{stem}_distance.csv"
                    pd.DataFrame(
                        abundance,
                        index=pd.Index(clone_ids, name="clone_id"),
                        columns=["Day0", "Day4", "Day9", "Day15"],
                    ).to_csv(input_path)

                    process = subprocess.run(
                        [
                            str(RSCRIPT),
                            str(R_SCRIPT),
                            str(input_path),
                            "6",
                            str(labels_path),
                            str(correlation_path),
                            str(distance_path),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    python_failure = None
                    try:
                        python_labels, python_correlation = (
                            cyclone_adapter.cyclone_correlation_core(counts, 6)
                        )
                    except cyclone_adapter.legacy_core.NativeSelectionFailure as exc:
                        python_failure = str(exc)

                    if process.returncode != 0:
                        self.assertNotEqual(process.returncode, 0)
                        self.assertIsNotNone(python_failure)
                        continue

                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertIsNone(python_failure)
                    r_labels = pd.read_csv(labels_path)
                    r_correlation = pd.read_csv(
                        correlation_path, index_col=0
                    ).to_numpy(dtype=float)
                    r_distance = pd.read_csv(
                        distance_path, index_col=0
                    ).to_numpy(dtype=float)
                    python_distance = (
                        cyclone_adapter.legacy_core.cyclone_r_pairwise_euclidean(
                            python_correlation
                        )
                    )

                    self.assertEqual(
                        r_labels["clone_id"].astype(str).tolist(),
                        clone_ids.astype(str).tolist(),
                    )
                    np.testing.assert_array_equal(
                        np.isnan(r_correlation), np.isnan(python_correlation)
                    )
                    np.testing.assert_allclose(
                        r_correlation,
                        python_correlation,
                        atol=1e-10,
                        rtol=0,
                        equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        r_distance, python_distance, atol=1e-10, rtol=0
                    )
                    r_values = r_labels["label"].to_numpy()
                    self.assertEqual(
                        adjusted_rand_score(r_values, python_labels), 1.0
                    )
                    np.testing.assert_array_equal(
                        r_values[:, None] == r_values[None, :],
                        python_labels[:, None] == python_labels[None, :],
                    )

    @unittest.skipUnless(RSCRIPT, "Rscript unavailable")
    def test_pairwise_correlation_intermediate_matches_r_exactly(self):
        matrix = np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [-1.0, 1.0, -2.0, 2.0],
                [0.0, 2.0, -1.0, 3.0],
                [1.0, 3.0, 0.0, 4.0],
            ]
        )
        python_correlation = (
            cyclone_adapter.r_pairwise_complete_correlation(matrix)
        )
        with tempfile.TemporaryDirectory(prefix="cyclone_cor_validation_") as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / "matrix.csv"
            output_path = tmpdir / "correlation.csv"
            pd.DataFrame(matrix).to_csv(input_path, index=False, header=False, na_rep="NA")
            expression = (
                "x<-as.matrix(read.csv(commandArgs(TRUE)[1],header=FALSE," 
                "na.strings='NA'));"
                "write.csv(cor(x,method='pearson',use='pairwise.complete.obs'),"
                "commandArgs(TRUE)[2],row.names=FALSE)"
            )
            process = subprocess.run(
                [str(RSCRIPT), "-e", expression, str(input_path), str(output_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            r_correlation = pd.read_csv(output_path).to_numpy(dtype=float)
        np.testing.assert_allclose(
            r_correlation,
            python_correlation,
            atol=1e-15,
            rtol=0,
            equal_nan=True,
        )

    @unittest.skipUnless(RSCRIPT and R_SCRIPT.is_file(), "Rscript unavailable")
    def test_finite_synthetic_tensor_matches_r_intermediates_and_partition(self):
        from sklearn.metrics import adjusted_rand_score

        counts, clone_ids = finite_synthetic_counts()
        abundance = counts.sum(axis=2, dtype=np.int64)
        python_labels, python_correlation = (
            cyclone_adapter.cyclone_correlation_core(counts, 6)
        )
        python_distance = (
            cyclone_adapter.legacy_core.cyclone_r_pairwise_euclidean(
                python_correlation
            )
        )
        with tempfile.TemporaryDirectory(prefix="cyclone_finite_validation_") as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / "abundance.csv"
            labels_path = tmpdir / "labels.csv"
            correlation_path = tmpdir / "correlation.csv"
            distance_path = tmpdir / "distance.csv"
            pd.DataFrame(
                abundance,
                index=pd.Index(clone_ids, name="clone_id"),
                columns=["Day0", "Day4", "Day9", "Day15"],
            ).to_csv(input_path)
            process = subprocess.run(
                [
                    str(RSCRIPT),
                    str(R_SCRIPT),
                    str(input_path),
                    "6",
                    str(labels_path),
                    str(correlation_path),
                    str(distance_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            r_labels = pd.read_csv(labels_path)
            r_correlation = pd.read_csv(
                correlation_path, index_col=0
            ).to_numpy(dtype=float)
            r_distance = pd.read_csv(
                distance_path, index_col=0
            ).to_numpy(dtype=float)
        np.testing.assert_allclose(
            r_correlation, python_correlation, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            r_distance, python_distance, atol=1e-10, rtol=0
        )
        r_values = r_labels["label"].to_numpy()
        self.assertEqual(adjusted_rand_score(r_values, python_labels), 1.0)
        np.testing.assert_array_equal(
            r_values[:, None] == r_values[None, :],
            python_labels[:, None] == python_labels[None, :],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
