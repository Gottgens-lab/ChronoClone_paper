#!/usr/bin/env python3
"""Build the visible, executable notebook for the corrected preflight gate."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "0001_day0_locked_setup_and_preflight.ipynb"


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "chronoclone",
            "language": "python",
            "name": "chronoclone",
        },
        "language_info": {"name": "python", "version": "3.11.15"},
        "execution": {
            "environment_name": "chronoclone",
            "environment_path": "/path/to/ChronoClone/.runtime/chronoclone",
            "kernel": "chronoclone",
            "command": (
                "/path/to/ChronoClone/.runtime/chronoclone/bin/python "
                "scripts/execute_notebook.py notebooks/0001_day0_locked_setup_and_preflight.ipynb"
            ),
            "status": "not_executed",
        },
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Fixed-t1 benchmark: setup and structural preflight\n\n"
            "This notebook verifies the corrected order before any production fit. "
            "Computed findings appear only in code outputs."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, platform\n"
            "import pandas as pd\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "assert ROOT.name == 'chronoclone_method_noise_benchmark_day0_locked_20260812'\n"
            "ENVIRONMENT = {\n"
            "    'name': 'chronoclone',\n"
            "    'path': '/path/to/ChronoClone/.runtime/chronoclone',\n"
            "    'kernel': 'chronoclone',\n"
            "    'python': platform.python_version(),\n"
            "    'execution_command': '/path/to/ChronoClone/.runtime/chronoclone/bin/python scripts/execute_notebook.py notebooks/0001_day0_locked_setup_and_preflight.ipynb',\n"
            "}\n"
            "assert ENVIRONMENT['python'] == '3.11.15'\n"
            "ENVIRONMENT"
        ),
        nbf.v4.new_code_cell(
            "report_path = ROOT / 'results/preflight/structural_report.json'\n"
            "table_path = ROOT / 'results/preflight/structural_conditions.csv'\n"
            "report = json.loads(report_path.read_text())\n"
            "conditions = pd.read_csv(table_path, dtype={'gaussian_seed': str, 'uniform_seed': str})\n"
            "assert report['status'] == 'pass'\n"
            "assert report['n_conditions'] == 243\n"
            "assert len(conditions) == 243\n"
            "report_display = dict(report)\n"
            "report_display['t1_changed_entries_total'] = report_display.pop('day0_changed_entries_total')\n"
            "report_display['t1_forbidden_state_count_total'] = report_display.pop('day0_forbidden_state_count_total')\n"
            "pd.DataFrame([report_display]).T.rename(columns={0: 'value'})"
        ),
        nbf.v4.new_code_cell(
            "gates = pd.DataFrame({\n"
            "    'gate': [\n"
            "        'clean eligibility retains 517',\n"
            "        't1 has zero changed entries',\n"
            "        't1 has zero forbidden-state cells',\n"
            "        'uniform does not change eligibility',\n"
            "        'uniform preserves clone-time totals',\n"
            "        'all 243 structural coordinates are unique',\n"
            "    ],\n"
            "    'pass': [\n"
            "        report['clean_eligible_clones'] == 517,\n"
            "        report['day0_changed_entries_total'] == 0,\n"
            "        report['day0_forbidden_state_count_total'] == 0,\n"
            "        report['uniform_eligibility_changes_total'] == 0,\n"
            "        report['uniform_clone_time_total_changes_total'] == 0,\n"
            "        not conditions.duplicated(['repeat','gaussian_pct','uniform_pct']).any(),\n"
            "    ],\n"
            "})\n"
            "assert gates['pass'].all()\n"
            "gates"
        ),
        nbf.v4.new_code_cell(
            "eligibility_summary = (conditions.groupby(['gaussian_pct','uniform_pct'], as_index=False)\n"
            "    .agg(n_pipeline_eligible_mean=('n_pipeline_eligible','mean'),\n"
            "         n_pipeline_eligible_min=('n_pipeline_eligible','min'),\n"
            "         n_pipeline_eligible_max=('n_pipeline_eligible','max')))\n"
            "eligibility_summary.head(20)"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation boundary\n\n"
            "Passing this notebook proves structural pipeline fidelity, not method "
            "accuracy. The behavioral fit sentinel and production result validation "
            "remain separate gates."
        ),
    ]
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, DESTINATION)
    print(DESTINATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
