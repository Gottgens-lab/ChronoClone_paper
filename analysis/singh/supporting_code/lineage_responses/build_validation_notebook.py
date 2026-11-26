"""Build a separate execution and artifact-integrity audit notebook."""
from pathlib import Path
import nbformat as nb

O=Path(__file__).resolve().parent
book=nb.v4.new_notebook(metadata={'kernelspec':{'name':'chronoclone','display_name':'chronoclone','language':'python'}})
book.cells=[nb.v4.new_markdown_cell('''# Execution and artifact-integrity audit

Read-only validation of the completed lineage-state analysis and its separately inspected seven PNGs. This adds no scientific hypothesis tests and does not change plots or source data. Environment `chronoclone` at `/path/to/ChronoClone/.runtime/chronoclone`, kernel `chronoclone`; versions and absolute execution command are displayed below. Run this notebook after `0001_lineage_state_responses.ipynb`.
'''),nb.v4.new_code_cell(r'''
from pathlib import Path
import sys,json,hashlib,platform,importlib.metadata as metadata
import nbformat,pandas as pd
from IPython.display import display
ROOT=Path('/path/to/ChronoClone')
O=ROOT/'analysis_workspace/singh/singh_lineage_state_responses_20260911'
ENV=Path('/path/to/ChronoClone/.runtime/chronoclone')
assert Path(sys.executable).parent.parent==ENV
environment={'name':'chronoclone','path':str(ENV),'kernel':'chronoclone','Python':platform.python_version(),'versions':{p:metadata.version(p) for p in ['nbformat','nbclient','numpy','pandas','scipy','statsmodels','matplotlib']},'execution_command':f'{ENV}/bin/python {O}/execute_notebook.py {O}/0002_execution_validation.ipynb'}
display(environment)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
analysis=nbformat.read(O/'0001_lineage_state_responses.ipynb',as_version=4)
code=[c for c in analysis.cells if c.cell_type=='code']
errors=[(i,o) for i,c in enumerate(code) for o in c.get('outputs',[]) if o.output_type=='error']
stderr=[(i,o.text) for i,c in enumerate(code) for o in c.get('outputs',[]) if o.output_type=='stream' and o.name=='stderr' and o.text]
assert not errors and not stderr
assert [c.execution_count for c in code]==list(range(1,len(code)+1))
sources=pd.read_csv(O/'reports/source_manifest.csv')
outputs=pd.read_csv(O/'reports/output_manifest.csv')
for table in [sources,outputs]:
    assert table.path.is_unique
    for row in table.itertuples():assert sha(ROOT/row.path)==row.sha256
reviews=json.loads((O/'reports/visual_review_decisions.json').read_text())['panels']
assert len(reviews)==7 and len({r['path'] for r in reviews})==7
for row in reviews:
    assert row['status']=='pass' and row['individually_viewed'] and sha(ROOT/row['path'])==row['sha256']
pngs=set(str(p.relative_to(ROOT)) for p in (O/'panels').glob('*.png'))
assert pngs=={r['path'] for r in reviews}
summary=json.loads((O/'reports/results_summary.json').read_text())
assert summary['visual_review']=='pass' and summary['planned_tests']==summary['BH_family']==340
report={'status':'pass','analysis_notebook':str((O/'0001_lineage_state_responses.ipynb').relative_to(ROOT)),'analysis_notebook_sha256':sha(O/'0001_lineage_state_responses.ipynb'),'executed_code_cells':len(code),'errors':len(errors),'stderr_streams':len(stderr),'source_hashes_verified':len(sources),'output_hashes_verified':len(outputs),'individually_reviewed_panels':len(reviews),'review_json_sha256':sha(O/'reports/visual_review_decisions.json'),'handoff_sha256':sha(O/'reports/RESULTS_AND_CAPTIONS.md'),'environment':environment}
(O/'reports/execution_validation.json').write_text(json.dumps(report,indent=2))
display(report)
''')]
nb.write(book,O/'0002_execution_validation.ipynb')
print(O/'0002_execution_validation.ipynb')
