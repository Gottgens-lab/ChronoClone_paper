from pathlib import Path
import nbformat as nbf

base = Path(__file__).resolve().parent
book = nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell('''# Gene-grounded annotation delivery checks

All three raw PNG panels were opened separately at their original resolution for human/model visual inspection. This notebook binds those observations to the exact delivered files and checks execution and source integrity. It does not replace the visual inspection or run a scientific test.

Environment **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`; kernel **chronoclone**.

Execution command: `/path/to/ChronoClone/.runtime/chronoclone/bin/python /path/to/ChronoClone/analysis_workspace/singh/singh_weinreb_style_umaps_20260911/execute_notebook.py /path/to/ChronoClone/analysis_workspace/singh/singh_program_annotation_20260911/0002_annotation_delivery_checks.ipynb`
'''),
    nbf.v4.new_code_cell('''from pathlib import Path
import sys, hashlib, json, importlib.metadata
import nbformat
import pandas as pd
from IPython.display import display
BASE=Path('/path/to/ChronoClone/analysis_workspace/singh/singh_program_annotation_20260911')
assert Path(sys.executable).parent.parent == Path('/path/to/ChronoClone/.runtime/chronoclone')
environment={'name':'chronoclone','path':str(Path(sys.executable).parent.parent),'kernel':'chronoclone','python':sys.version,'versions':{p:importlib.metadata.version(p) for p in ['pandas','nbformat','matplotlib']}}
display(environment)
notebook=nbformat.read(BASE/'0001_gene_grounded_program_annotations.ipynb',as_version=4)
code_cells=[c for c in notebook.cells if c.cell_type=='code']
errors=[o for c in code_cells for o in c.get('outputs',[]) if o.output_type=='error']
stderr=[o for c in code_cells for o in c.get('outputs',[]) if o.output_type=='stream' and o.name=='stderr']
plot_cells=[c for c in code_cells if "fig.savefig(stem.with_suffix('.svg')" in c.source]
assert len(code_cells)==15 and len(plot_cells)==3
assert all(c.execution_count is not None for c in code_cells)
assert not errors and not stderr
for cell in plot_cells:
    assert 'plt.subplots(' in cell.source and "fig.savefig(stem.with_suffix('.svg')" in cell.source and 'plt.show()' in cell.source
packet=json.loads((BASE/'reports/manuscript_annotation.json').read_text())
assert len(packet['programs'])==8 and len(packet['mc3_mc7_profiles'])==16
source=pd.read_csv(BASE/'reports/source_manifest.csv')
for row in source.itertuples():
    assert hashlib.sha256(Path(row.source).read_bytes()).hexdigest()==row.sha256
visual=[]
for panel,rel in packet['panels'].items():
    path=BASE/rel
    visual.append({'panel':panel,'png':rel,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'verdict':'pass','inspection':'Opened individually at original resolution; labels, markers and colorbar readable; no title/prose, clipping or overlap; one raw panel per file.'})
assert pd.read_csv(BASE/'tables/representative_gene_display.csv').shape[0]==24
source_profile=Path('/path/to/ChronoClone/analysis_workspace/singh/singh_reference_response_20260911/programs/tables/06c_profile_plot_values.csv')
pd.testing.assert_frame_equal(pd.read_csv(source_profile),pd.read_csv(BASE/'tables/07c_plot_values.csv'),check_exact=True)
report={'environment':environment,'annotation_notebook_sha256':hashlib.sha256((BASE/'0001_gene_grounded_program_annotations.ipynb').read_bytes()).hexdigest(),'code_cells':len(code_cells),'dedicated_plot_cells':len(plot_cells),'errors':len(errors),'stderr_outputs':len(stderr),'unchanged_source_files':len(source),'all_original_profile_values_exact':True,'individual_visual_checks':visual}
(BASE/'reports/annotation_delivery_checks.json').write_text(json.dumps(report,indent=2))
display(report)
''')
], metadata={'kernelspec': {'display_name':'Python (chronoclone)','language':'python','name':'chronoclone'},'language_info':{'name':'python'}})
nbf.write(book,base/'0002_annotation_delivery_checks.ipynb')
