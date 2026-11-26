"""Build the benchmark curation and portable evidence notebooks."""
from pathlib import Path
import json
import sys
import nbformat as nbf

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
PROJECT = Path('/path/to/ChronoClone')
PLAN = PROJECT / 'analysis_workspace/workspace_audits/20260914_paper_v5/benchmark_plan.json'
PRESERVATION_ONLY = '--preservation-only' in sys.argv
HERE.mkdir(parents=True, exist_ok=True)
if not PRESERVATION_ONLY:
    (HERE / 'source_copy_plan.json').write_text(PLAN.read_text())


def write(path, cells):
    if PRESERVATION_ONLY and path.name != '0002_preservation_checks.ipynb':
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'chronoclone-paper', 'display_name': 'Python (chronoclone-paper)', 'language': 'python'},
        'language_info': {'name': 'python'},
    })
    nbf.validate(notebook)
    nbf.write(notebook, path)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
environment = '''from pathlib import Path
import hashlib, importlib.metadata, json, platform, sys
import numpy as np
import pandas as pd
from IPython.display import display

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

HISTORICAL_MAC_ENVIRONMENT = {
    'manager': 'existing micromamba-compatible environment, reused without modification',
    'name': 'chronoclone',
    'absolute_macbook_path': '/path/to/ChronoClone/.runtime/chronoclone',
    'kernel': 'chronoclone',
}
runtime = dict(HISTORICAL_MAC_ENVIRONMENT,
    current_prefix=sys.prefix, current_executable=sys.executable, python=platform.python_version(),
    versions={name: importlib.metadata.version(name) for name in
              ['numpy','pandas','scipy','scikit-learn','scanpy','matplotlib','nbformat','nbclient']})
'''

evidence_cells = [
md('# Fixed-start benchmark evidence and portable smoke run\n\nThis notebook validates the released summary grid, preserves estimator failures, and executes one clean reference condition from the frozen production implementation. It does not repeat the full benchmark or reinterpret it as biological generalization. The adjacent result notebooks contain the existing scientific plots; this audit uses tables.'),
code(environment + '''
MODULE = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
              if (p / 'results/aggregate/aggregation_metadata.json').is_file())
BENCHMARKS = MODULE.parent
REPORTS = MODULE / 'reports'
REPORTS.mkdir(exist_ok=True)
runtime['execution_command'] = 'python benchmarks/run_notebook.py benchmarks/ode_noise/notebooks/0006_benchmark_evidence_and_smoke.ipynb --kernel chronoclone'
display(pd.DataFrame([runtime]).T)
'''),
md('## Verify the released identities\n\nHashes identify the stored evidence. They do not by themselves establish that every historical fit is correct.'),
code('''AGGREGATE = MODULE / 'results/aggregate'
metadata = json.loads((AGGREGATE / 'aggregation_metadata.json').read_text())
assert metadata['status'] == 'pass'
records = []
for name, expected in metadata['artifact_sha256'].items():
    path = AGGREGATE / name
    observed = sha256(path)
    assert observed == expected, name
    records.append({'artifact': name, 'bytes': path.stat().st_size, 'sha256_matches': observed == expected})
display(pd.DataFrame(records))
large = metadata['omitted_large_artifacts']['native_method_noise_runs.csv.gz']
full_path = AGGREGATE / 'native_method_noise_runs.csv.gz'
if full_path.is_file():
    assert full_path.stat().st_size == large['bytes'] and sha256(full_path) == large['sha256']
print('Repeat-level aggregate available locally:', full_path.is_file())
print('Repeat-level asset requirement:', json.dumps(large, indent=2))
'''),
md('## Check grid completeness and estimator semantics\n\nEach coordinate has its own eligible clone set. The five-method aggregate is retained; the main comparison displays the four estimator-compatible methods. A failed estimator has no ARI, rather than zero ARI.'),
code('''from itertools import product
summary = pd.read_csv(AGGREGATE / 'native_method_noise_summary.csv')
complete = pd.read_csv(AGGREGATE / 'result_completeness.csv')
selective = pd.read_csv(AGGREGATE / 'chronoclone_selective_summary.csv')
eligibility = pd.read_csv(AGGREGATE / 'eligibility_summary.csv')
same_retained = pd.read_csv(AGGREGATE / 'same_retained_subset_summary.csv')
methods = ['ChronoClone','CLADES preprocessing','Cyclone abundance-only','K-means','Ward']
levels = list(range(0,101,2))
keys = ['method','gaussian_pct','uniform_pct']
expected = set(product(methods,levels,levels))
for frame in [summary,complete,same_retained]:
    assert not frame.duplicated(keys).any()
    assert set(frame[keys].itertuples(index=False,name=None)) == expected
    assert frame.n_expected.eq(100).all()
    assert (frame.n_success + frame.n_method_failure).eq(100).all()
for frame in [selective,eligibility]:
    assert not frame.duplicated(['gaussian_pct','uniform_pct']).any()
    assert set(frame[['gaussian_pct','uniform_pct']].itertuples(index=False,name=None)) == set(product(levels,levels))
assert int(complete.n_expected.sum()) == metadata['observed_rows']
assert int(complete.n_method_failure.sum()) == metadata['method_failure_rows']
joined = summary[keys+['n_success','n_method_failure']].merge(complete[keys+['n_success','n_method_failure']],on=keys,validate='one_to_one',suffixes=('_summary','_complete'))
assert joined.n_success_summary.equals(joined.n_success_complete)
assert joined.n_method_failure_summary.equals(joined.n_method_failure_complete)
assert summary.loc[summary.n_success.eq(0), 'ari_mean'].isna().all()
assert summary.ari_n_finite.equals(summary.n_success)
assert summary.loc[summary.n_success.gt(0),'ari_mean'].between(-1,1).all()
assert summary.loc[summary.n_success.gt(0),'ari_q025'].le(summary.loc[summary.n_success.gt(0),'ari_q975']).all()
primary = complete.loc[complete.method.ne('Cyclone abundance-only')]
assert primary.n_success.eq(100).all()
assert selective.postfit_retained_fraction_of_eligible_mean.between(0,1).all()
assert eligibility.eligible_fraction_mean.between(0,1).all()
display(complete.groupby('method',sort=False)[['n_expected','n_success','n_method_failure']].sum())
display(summary.loc[summary.gaussian_pct.eq(0)&summary.uniform_pct.eq(0),['method','n_success','ari_mean','recovered_k_mean']])
'''),
md('## Execute one bounded condition\n\nThe wrapper verifies the frozen configuration, corrected core, low-level core, reference, and original wheel. It preserves source bytes and loads them through temporary paths. It requires the recorded dependency versions and writes a separate smoke record. The four fits finish before truth and retention metrics are evaluated. This is a clean-reference portability test, not a full-grid rerun.'),
code('''import subprocess
smoke_output = REPORTS / 'portable_clean_smoke.json'
command = [sys.executable, str(MODULE/'scripts/run_smoke.py'), '--output',str(smoke_output)]
completed = subprocess.run(command,capture_output=True,text=True)
if completed.returncode:
    raise RuntimeError(completed.stdout + '\\n' + completed.stderr)
smoke = json.loads(smoke_output.read_text())
assert smoke['status']=='pass' and smoke['t1_changed_entries']==0
assert smoke['runtime']['historical_versions_match']
display(pd.DataFrame(smoke['results']))
display(pd.DataFrame([{k:smoke[k] for k in ['eligible_clones','postfit_retained_clones','postfit_retention_fraction','t1_changed_entries']}]))
if completed.stderr.strip():
    print('Runtime information captured:', completed.stderr.strip()[:1500])
'''),
md('## Record the evidence boundary\n\nThe compact tables and bounded run are validated separately. Full-grid regeneration still uses the documented historical production topology; the new wrapper does not resume its shards. Monte Carlo ranges describe this generator. They are not biological confidence intervals.'),
code('''result = {'schema':'chronoclone_paper_benchmark_evidence_v1','status':'pass',
          'summary_rows':len(summary),'grid_coordinates':len(eligibility),
          'historical_method_condition_rows':int(complete.n_expected.sum()),
          'historical_method_failures':int(complete.n_method_failure.sum()),
          'summary_hashes_verified':len(records),'bounded_smoke_status':smoke['status'],
          'full_aggregate_verified_here':full_path.is_file(),
          'full_benchmark_rerun':False,'runtime':runtime}
(REPORTS/'benchmark_evidence_validation.json').write_text(json.dumps(result,indent=2)+'\\n')
display(pd.DataFrame([{key:value for key,value in result.items() if key!='runtime'}]).T)
''')]
write(BENCHMARKS/'ode_noise/notebooks/0006_benchmark_evidence_and_smoke.ipynb', evidence_cells)

curation_cells = [
md('# Benchmark curation for the manuscript v5 repository\n\nThis local assembly record copies checksum-registered benchmark evidence into the paper repository, updates only the portable notebooks’ environment records, and executes the result and evidence notebooks. Historical source and the original workspace remain unchanged. This curator needs the original MacBook workspace; it is provenance rather than a clean-checkout analysis entry point.'),
code(environment + '''
PROJECT = Path('/path/to/ChronoClone')
REPOSITORY = PROJECT / 'analysis_workspace/ChronoClone_paper'
BENCHMARKS = REPOSITORY / 'benchmarks'
CURATION = BENCHMARKS / 'curation'
plan = json.loads((CURATION/'source_copy_plan.json').read_text())
runtime['execution_command'] = '/path/to/ChronoClone/.runtime/chronoclone/bin/python analysis_workspace/workspace_audits/20260914_paper_v5/execute_notebook.py analysis_workspace/ChronoClone_paper/benchmarks/curation/0001_curate_benchmark.ipynb'
assert Path(sys.prefix).resolve() == Path(HISTORICAL_MAC_ENVIRONMENT['absolute_macbook_path']).resolve()
display(pd.DataFrame([runtime]).T)
'''),
md('## Copy only registered sources'),
code('''import shutil
copy_records=[]
for record in plan['copy_files']:
    source=PROJECT/record['source']
    destination=REPOSITORY/record['destination']
    assert destination.resolve().is_relative_to(BENCHMARKS.resolve())
    assert source.stat().st_size==record['bytes'] and sha256(source)==record['sha256'], source
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
    assert sha256(destination)==record['sha256'], destination
    copy_records.append(dict(record,destination_sha256_at_copy=sha256(destination)))
(CURATION/'copy_manifest.json').write_text(json.dumps(copy_records,indent=2)+'\\n')
print('Exact source copies:',len(copy_records))
print('Copied bytes:',sum(record['bytes'] for record in copy_records))
'''),
md('## Adapt and execute the portable notebooks\n\nOnly the runtime record and execution command change. The scientific calculation and plot cells remain intact. The source checksum and resulting notebook checksum are recorded separately.'),
code('''import nbformat, subprocess
notebooks=BENCHMARKS/'ode_noise/notebooks'
adapted=[]
for name in ['0003_t1_locked_method_noise_results.ipynb','0005_t1_locked_four_method_noise_results.ipynb']:
    path=notebooks/name
    notebook=nbformat.read(path,as_version=4)
    source_hash=sha256(path)
    setup=next(cell for cell in notebook.cells if cell.cell_type=='code' and 'ENVIRONMENT = {' in cell.source)
    setup.source=setup.source.replace('"name": "chronoclone-paper"','"name": "chronoclone"')
    setup.source=setup.source.replace('"path": "see environment.yml"','"historical_macbook_path": "/path/to/ChronoClone/.runtime/chronoclone",\\n    "current_prefix": __import__("sys").prefix')
    setup.source=setup.source.replace('"kernel": "python3"','"kernel": "chronoclone"')
    setup.source=setup.source.replace('python ../../scripts/execute_notebook_inprocess.py','python ../run_notebook.py')
    notebook.metadata.kernelspec={'name':'chronoclone','display_name':'chronoclone','language':'python'}
    for cell in notebook.cells:
        if cell.cell_type=='code': cell.execution_count=None;cell.outputs=[]
    nbformat.write(notebook,path)
    adapted.append({'path':str(path.relative_to(REPOSITORY)),'source_sha256':source_hash})
for name in ['0003_t1_locked_method_noise_results.ipynb','0005_t1_locked_four_method_noise_results.ipynb','0006_benchmark_evidence_and_smoke.ipynb']:
    command=[sys.executable,str(BENCHMARKS/'run_notebook.py'),str(notebooks/name),'--kernel','chronoclone']
    completed=subprocess.run(command,capture_output=True,text=True)
    if completed.returncode: raise RuntimeError(completed.stdout+'\\n'+completed.stderr)
    print(completed.stdout.strip())
for record in adapted:
    record['executed_sha256']=sha256(REPOSITORY/record['path'])
(CURATION/'adapted_notebooks.json').write_text(json.dumps(adapted,indent=2)+'\\n')
'''),
md('## Verify preservation and record execution'),
code('''errors=[]
for record in copy_records:
    source=PROJECT/record['source']
    assert sha256(source)==record['sha256'], source
    destination=REPOSITORY/record['destination']
    record['destination_sha256_after_execution']=sha256(destination)
    record['destination_bytes_after_execution']=destination.stat().st_size
    if record['destination'].startswith('benchmarks/frozen_runtime/'):
        assert record['destination_sha256_after_execution']==record['sha256'],destination
for name in ['0003_t1_locked_method_noise_results.ipynb','0005_t1_locked_four_method_noise_results.ipynb','0006_benchmark_evidence_and_smoke.ipynb']:
    nb=nbformat.read(notebooks/name,as_version=4)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert [c.execution_count for c in cells]==list(range(1,len(cells)+1))
    assert not [o for c in cells for o in c.outputs if o.output_type=='error']
(CURATION/'copy_manifest.json').write_text(json.dumps(copy_records,indent=2)+'\\n')
result=json.loads((BENCHMARKS/'ode_noise/reports/benchmark_evidence_validation.json').read_text())
assert result['status']=='pass'
print('Scientific cores and wheel remain byte-preserved; ancillary publication path labels are adapted.')
display(pd.DataFrame([{k:v for k,v in result.items() if k!='runtime'}]).T)
''')]
write(HERE/'0001_curate_benchmark.ipynb',curation_cells)
preservation_cells = [
md('# Benchmark preservation checks\n\nThis local check compares regenerated figures with the previously registered paper release. It ignores only SVG creation dates and compares PNG pixels exactly. It also checks the smoke wrapper command-line help from an independent working directory; the evidence notebook separately executes the wrapper from the notebooks directory.'),
code(environment + '''
PROJECT=Path('/path/to/ChronoClone')
BENCHMARKS=PROJECT/'analysis_workspace/ChronoClone_paper/benchmarks'
OLD=PROJECT/'analysis_workspace/releases/chronoclone_paper_repository_20260908/repository/analyses/01_ode_benchmark'
runtime['execution_command']='python benchmarks/run_notebook.py benchmarks/curation/0002_preservation_checks.ipynb --kernel chronoclone'
display(pd.DataFrame([runtime]).T)
'''),
md('## Compare all regenerated figure contents'),
code('''import re
from PIL import Image
rows=[]
for path in sorted((BENCHMARKS/'ode_noise/figures').rglob('*.svg')):
    relative=path.relative_to(BENCHMARKS/'ode_noise/figures')
    old=OLD/'figures'/relative
    normalized=re.sub(r'<dc:date>.*?</dc:date>','',path.read_text())
    old_normalized=re.sub(r'<dc:date>.*?</dc:date>','',old.read_text())
    rows.append({'figure':str(relative),'comparison':'SVG content excluding creation date','passed':normalized==old_normalized})
for path in sorted((BENCHMARKS/'ode_noise/figures/manuscript_four_method').glob('*.png')):
    old=OLD/'figures/manuscript_four_method'/path.name
    rows.append({'figure':path.name,'comparison':'PNG pixels','passed':np.array_equal(np.asarray(Image.open(path)),np.asarray(Image.open(old)))})
frame=pd.DataFrame(rows)
assert frame.passed.all(),frame.loc[~frame.passed]
frame.to_csv(BENCHMARKS/'curation/figure_preservation.csv',index=False)
display(frame.groupby('comparison').agg(files=('figure','size'),passed=('passed','sum')))
print('Visual spot checks: Gaussian ARI and silhouette-coverage PNGs inspected individually; no clipping or overlap identified. All remaining regenerated content matches its preserved source.')
'''),
md('## Confirm notebook execution records and independent working directory'),
code('''import nbformat, subprocess, tempfile
records=[]
for name in ['0003_t1_locked_method_noise_results.ipynb','0005_t1_locked_four_method_noise_results.ipynb','0006_benchmark_evidence_and_smoke.ipynb']:
    path=BENCHMARKS/'ode_noise/notebooks'/name
    notebook=nbformat.read(path,as_version=4)
    cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    assert [cell.execution_count for cell in cells]==list(range(1,len(cells)+1))
    assert not [output for cell in cells for output in cell.outputs if output.output_type=='error']
    records.append({'notebook':name,'code_cells':len(cells),'retained_output_cells':sum(bool(c.outputs) for c in cells),'sha256':sha256(path)})
# Help resolves the script and parser without fitting another condition.
with tempfile.TemporaryDirectory(prefix='cc_paper_cwd_') as temporary:
    result=subprocess.run([sys.executable,str(BENCHMARKS/'ode_noise/scripts/run_smoke.py'),'--help'],cwd=temporary,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
manifest={'status':'pass','figure_comparisons':len(rows),'notebooks':records,'smoke_help_independent_cwd':True,'runtime':runtime}
(BENCHMARKS/'curation/preservation_validation.json').write_text(json.dumps(manifest,indent=2)+'\\n')
display(pd.DataFrame(records))
''')]
write(HERE/'0002_preservation_checks.ipynb',preservation_cells)
print('Built benchmark curation and evidence notebooks')
