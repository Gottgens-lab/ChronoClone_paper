#!/usr/bin/env python3
"""Build the visible behavioral-pilot validation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "0002_day0_locked_behavioral_pilot.ipynb"


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
                "scripts/execute_notebook.py notebooks/0002_day0_locked_behavioral_pilot.ipynb"
            ),
            "status": "not_executed",
        },
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Fixed-t1 behavioral pilot\n\n"
            "This notebook validates the three-repeat low-noise fitting pilot and "
            "records the corrected interpretation boundary before production."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import glob, hashlib, json, platform, subprocess\n"
            "import pandas as pd\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "ENVIRONMENT = {\n"
            " 'name': 'chronoclone',\n"
            " 'path': '/path/to/ChronoClone/.runtime/chronoclone',\n"
            " 'kernel': 'chronoclone',\n"
            " 'python': platform.python_version(),\n"
            " 'execution_command': '/path/to/ChronoClone/.runtime/chronoclone/bin/python scripts/execute_notebook.py notebooks/0002_day0_locked_behavioral_pilot.ipynb',\n"
            "}\n"
            "assert ENVIRONMENT['python'] == '3.11.15'\n"
            "ENVIRONMENT"
        ),
        nbf.v4.new_code_cell(
            "paths = sorted((ROOT/'results/preflight/behavioral_chunks').glob('*.csv.gz'))\n"
            "assert len(paths) == 3\n"
            "runs = pd.concat([pd.read_csv(path, dtype={'gaussian_seed': str, 'uniform_seed': str}) for path in paths], ignore_index=True)\n"
            "assert len(runs) == 1080\n"
            "assert not runs.duplicated(['repeat','gaussian_pct','uniform_pct','method']).any()\n"
            "assert runs['day0_changed_entries'].eq(0).all()\n"
            "assert runs['day0_tensor_sha256'].eq(runs['clean_day0_tensor_sha256']).all()\n"
            "assert runs.groupby(['repeat','gaussian_pct','uniform_pct'])['eligible_clone_ids_sha256'].nunique().max() == 1\n"
            "runs.groupby(['method','status']).size().rename('rows').reset_index()"
        ),
        nbf.v4.new_code_cell(
            "cc = runs[(runs['method']=='ChronoClone') & (runs['status']=='ok')].copy()\n"
            "gaussian_axis = cc[cc['uniform_pct'].eq(0)].groupby('gaussian_pct', as_index=False).agg(ari_mean=('ari','mean'), recovered_k_mean=('recovered_k','mean'), eligible_mean=('n_eligible_clones','mean'), retained_mean=('n_postfit_retained_clones','mean'))\n"
            "uniform_axis = cc[cc['gaussian_pct'].eq(0)].groupby('uniform_pct', as_index=False).agg(ari_mean=('ari','mean'), recovered_k_mean=('recovered_k','mean'), eligible_mean=('n_eligible_clones','mean'), retained_mean=('n_postfit_retained_clones','mean'))\n"
            "assert gaussian_axis['ari_mean'].eq(1).all() and gaussian_axis['recovered_k_mean'].eq(12).all()\n"
            "assert uniform_axis['ari_mean'].eq(1).all() and uniform_axis['recovered_k_mean'].eq(12).all()\n"
            "pd.concat({'Gaussian-only': gaussian_axis.set_index('gaussian_pct'), 'Uniform-only': uniform_axis.set_index('uniform_pct')}, names=['axis','noise_pct'])"
        ),
        nbf.v4.new_code_cell(
            "gates = pd.DataFrame({\n"
            " 'gate': [\n"
            "  'three exact pilot shards', '1080 unique rows', 't1 immutable',\n"
            "  'common eligible IDs across methods', 'ChronoClone succeeds 216/216',\n"
            "  'Gaussian axis ARI 1 and K 12', 'Uniform axis ARI 1 and K 12',\n"
            "  'postfit cutoff exactly -0.1', 'no group-size or refit fields',\n"
            " ],\n"
            " 'pass': [\n"
            "  len(paths)==3, len(runs)==1080, runs['day0_changed_entries'].eq(0).all(),\n"
            "  runs.groupby(['repeat','gaussian_pct','uniform_pct'])['eligible_clone_ids_sha256'].nunique().max()==1,\n"
            "  len(cc)==216, gaussian_axis['ari_mean'].eq(1).all() and gaussian_axis['recovered_k_mean'].eq(12).all(),\n"
            "  uniform_axis['ari_mean'].eq(1).all() and uniform_axis['recovered_k_mean'].eq(12).all(),\n"
            "  cc['silhouette_cutoff'].eq(-0.1).all(),\n"
            "  not any(('group_size' in c or 'refit' in c) for c in runs.columns),\n"
            " ],\n"
            "})\n"
            "assert gates['pass'].all()\n"
            "gates"
        ),
        nbf.v4.new_code_cell(
            "test_command = [\n"
            " '/path/to/ChronoClone/.runtime/chronoclone/bin/python',\n"
            " '-m','unittest','-v','tests/test_cyclone_adapter.py'\n"
            "]\n"
            "test = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True, check=False)\n"
            "assert test.returncode == 0, test.stdout + test.stderr\n"
            "pd.DataFrame({'validation': ['Cyclone adapter unit tests'], 'status': ['pass']})"
        ),
        nbf.v4.new_code_cell(
            "cyclone = runs[runs['method'].eq('Cyclone abundance-only')]\n"
            "cyclone_summary = pd.DataFrame({\n"
            " 'pilot_rows': [len(cyclone)],\n"
            " 'successful_partitions': [int(cyclone['status'].eq('ok').sum())],\n"
            " 'method_failures': [int(cyclone['status'].eq('method_failure').sum())],\n"
            " 'interpretation': ['applicability/failure rate; no accuracy value when non-estimable'],\n"
            "})\n"
            "assert cyclone_summary.loc[0,'pilot_rows'] == 216\n"
            "cyclone_summary"
        ),
        nbf.v4.new_code_cell(
            "def sha256(path):\n"
            " h=hashlib.sha256()\n"
            " with open(path,'rb') as f:\n"
            "  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)\n"
            " return h.hexdigest()\n"
            "report = {\n"
            " 'status':'pass', 'behavioral_rows':len(runs), 'chronoclone_rows':len(cc),\n"
            " 'day0_changed_entries_total':int(runs['day0_changed_entries'].sum()),\n"
            " 'runner_sha256':sha256(ROOT/'scripts/run_chunk.py'),\n"
            " 'config_sha256':sha256(ROOT/'config/benchmark_config.json'),\n"
            " 'result_schema_sha256':sha256(ROOT/'scripts/result_schema.py'),\n"
            " 'cyclone_adapter_sha256':sha256(ROOT/'scripts/cyclone_adapter.py'),\n"
            " 'pilot_chunk_sha256':{path.name:sha256(path) for path in paths},\n"
            "}\n"
            "report_path=ROOT/'results/preflight/behavioral_report.json'\n"
            "report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\\n')\n"
            "report_display = dict(report)\n"
            "report_display['t1_changed_entries_total'] = report_display.pop('day0_changed_entries_total')\n"
            "report_display"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation boundary\n\n"
            "The three-repeat pilot demonstrates that the previous low-noise axis "
            "drop is absent after fixing t1 and applying pipeline eligibility. It does not "
            "replace the 100-repeat result. Cyclone failures are applicability "
            "outcomes, not zero accuracy."
        ),
    ]
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, DESTINATION)
    print(DESTINATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
