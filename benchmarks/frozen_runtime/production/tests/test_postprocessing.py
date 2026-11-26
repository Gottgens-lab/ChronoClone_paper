"""Unit tests for strict aggregation and one-plot-per-cell generation."""

from __future__ import annotations

import sys
import unittest
import importlib.util
import json
import io
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_local_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


local_result_schema = load_local_module(
    "day0_locked_postprocessing_result_schema", "result_schema.py"
)
# Source-runtime tests deliberately prepend the historical benchmark scripts.
# Pin this import name locally so discovery order cannot select that old schema.
sys.modules["result_schema"] = local_result_schema
METHODS = local_result_schema.METHODS
aggregate = load_local_module(
    "day0_locked_postprocessing_aggregate", "aggregate_results.py"
)
builder = load_local_module(
    "day0_locked_postprocessing_builder", "build_notebooks.py"
)


class AggregationHelpersTest(unittest.TestCase):
    def test_summary_excludes_nonfinite_values(self):
        observed = aggregate.summarize_vector(
            [0.0, 0.5, 1.0, np.nan], (0.025, 0.975)
        )
        self.assertEqual(observed["n_finite"], 3)
        self.assertEqual(observed["mean"], 0.5)
        self.assertAlmostEqual(observed["median"], 0.5)

    def test_empty_summary_is_explicit(self):
        observed = aggregate.summarize_vector([np.nan], (0.025, 0.975))
        self.assertEqual(observed["n_finite"], 0)
        self.assertTrue(np.isnan(observed["mean"]))

    def test_schema_version_canonicalization(self):
        self.assertEqual(aggregate.canonical_schema_version("3.0"), "3")
        self.assertEqual(aggregate.canonical_schema_version(3), "3")
        with self.assertRaises(ValueError):
            aggregate.canonical_schema_version("not-a-number")

    def test_common_condition_contract_detects_method_disagreement(self):
        base = {column: "locked" for column in aggregate.COMMON_CONDITION_COLUMNS}
        base.update({"repeat": 0, "gaussian_pct": 0, "uniform_pct": 0})
        table = pd.DataFrame([{**base, "method": method} for method in METHODS])
        aggregate.validate_common_condition_group(table)
        table.loc[0, "eligible_clone_ids_sha256"] = "changed"
        with self.assertRaises(ValueError):
            aggregate.validate_common_condition_group(table)

    def test_uniform_zero_diagnostics_must_be_zero(self):
        row = pd.Series(
            {
                "uniform_pct": 0,
                "uniform_requested_move_count": 0,
                "uniform_l1_half_count": 0,
                "uniform_changed_later_entries": 0,
            }
        )
        aggregate.validate_uniform_diagnostics(row)
        row["uniform_l1_half_count"] = 1
        with self.assertRaises(ValueError):
            aggregate.validate_uniform_diagnostics(row)

    def _valid_postfilter_group(self):
        mask = np.asarray([1, 0, 1], dtype=np.uint8)
        packed = np.packbits(mask, bitorder="little").tobytes().hex()
        base = {
            "status": "ok",
            "postfilter_status": "ok",
            "silhouette_cutoff": -0.1,
            "n_input_clones": 3,
            "n_eligible_clones": 3,
            "n_postfit_retained_clones": 2,
            "postfit_retained_fraction_of_eligible": 2 / 3,
            "postfit_retained_fraction_of_all_input": 2 / 3,
            "postfit_retention_mask_hex": packed,
            "postfit_retention_mask_sha256": local_result_schema.array_sha256(mask),
            "postfit_sample_silhouette_sha256": "a" * 64,
            "recovered_k": 2,
            "postfit_retained_k": 2,
            "postfit_clusters_lost": 0,
            "postfit_silhouette_min": -0.2,
            "postfit_silhouette_mean": 0.2,
            "postfit_silhouette_median": 0.3,
            "postfit_silhouette_max": 0.5,
            "program_retention_json": json.dumps({"MC01": 1.0, "MC02": 0.5}),
            "shared_retained_subset_ari": 0.8,
            "shared_retained_subset_nmi": 0.8,
            "shared_retained_subset_hungarian_macro_f1": 0.8,
            "shared_retained_subset_abstention_aware_hungarian_macro_f1": 0.6,
        }
        return pd.DataFrame([{**base, "method": method} for method in METHODS])

    def test_postfilter_payload_rejects_mask_and_metric_corruption(self):
        group = self._valid_postfilter_group()
        aggregate.validate_postfilter_condition(group, program_ids={"MC01", "MC02"})
        bad_metric = group.copy()
        bad_metric.loc[0, "shared_retained_subset_ari"] = 7
        with self.assertRaises(ValueError):
            aggregate.validate_postfilter_condition(
                bad_metric, program_ids={"MC01", "MC02"}
            )
        bad_mask = group.copy()
        bad_mask["postfit_retention_mask_hex"] = "00"
        with self.assertRaises(ValueError):
            aggregate.validate_postfilter_condition(
                bad_mask, program_ids={"MC01", "MC02"}
            )
        bad_fraction = group.copy()
        bad_fraction["postfit_retained_fraction_of_all_input"] = 2
        with self.assertRaises(ValueError):
            aggregate.validate_postfilter_condition(
                bad_fraction, program_ids={"MC01", "MC02"}
            )
        bad_program = group.copy()
        bad_program["program_retention_json"] = json.dumps({"MC01": 2.0})
        with self.assertRaises(ValueError):
            aggregate.validate_postfilter_condition(
                bad_program, program_ids={"MC01", "MC02"}
            )

    def test_chronoclone_method_failure_survives_csv_empty_field_roundtrip(self):
        base = {
            "status": "method_failure",
            "postfilter_status": "chronoclone_method_failure",
            "silhouette_cutoff": -0.1,
            "n_input_clones": 3,
            "n_eligible_clones": 3,
            "n_postfit_retained_clones": np.nan,
            "postfit_retained_fraction_of_eligible": np.nan,
            "postfit_retained_fraction_of_all_input": np.nan,
            "postfit_retained_k": np.nan,
            "postfit_clusters_lost": np.nan,
            "postfit_retention_mask_hex": "",
            "postfit_retention_mask_sha256": "",
            "postfit_sample_silhouette_sha256": "",
            "postfit_silhouette_min": np.nan,
            "postfit_silhouette_mean": np.nan,
            "postfit_silhouette_median": np.nan,
            "postfit_silhouette_max": np.nan,
            "program_retention_json": "{}",
            "shared_retained_subset_ari": np.nan,
            "shared_retained_subset_nmi": np.nan,
            "shared_retained_subset_hungarian_macro_f1": np.nan,
            "shared_retained_subset_abstention_aware_hungarian_macro_f1": np.nan,
        }
        group = pd.DataFrame([{**base, "method": method} for method in METHODS])
        buffer = io.StringIO()
        group.to_csv(buffer, index=False)
        buffer.seek(0)
        roundtripped = pd.read_csv(buffer)
        aggregate.validate_postfilter_condition(
            roundtripped, program_ids={"MC01", "MC02"}
        )


class NotebookSourceContractTest(unittest.TestCase):
    def test_each_generated_plot_cell_is_single_figure_single_svg(self):
        sources = [
            builder.axis_ari_cell("gaussian"),
            builder.axis_ari_cell("uniform"),
            *(builder.heatmap_cell(method) for method in METHODS),
            builder.axis_success_cell("gaussian"),
            builder.axis_success_cell("uniform"),
            builder.axis_k_cell("gaussian"),
            builder.axis_k_cell("uniform"),
            builder.axis_sensitivity_cell("gaussian"),
            builder.axis_sensitivity_cell("uniform"),
        ]
        self.assertEqual(len(sources), 13)
        for source in sources:
            self.assertEqual(source.count("plt.subplots("), 1)
            self.assertEqual(source.count("fig.savefig("), 1)
            self.assertEqual(source.count("plt.show()"), 1)
            self.assertNotIn("plt.subplots(", source.split("plt.subplots(", 1)[1])
        for method in METHODS:
            self.assertIn(
                "colorbar.solids.set_rasterized(False)",
                builder.heatmap_cell(method),
            )

    def test_no_arg_controller_entrypoints_exist(self):
        self.assertTrue((SCRIPTS / "aggregate_results.py").is_file())
        self.assertTrue((SCRIPTS / "build_notebooks.py").is_file())
        self.assertTrue((SCRIPTS / "validate_postprocessing.py").is_file())


if __name__ == "__main__":
    unittest.main()
