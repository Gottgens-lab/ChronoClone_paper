"""Create the reviewable presentation and manuscript-validation notebook.

This scaffold is documented in the notebook. It does not execute scientific
models or alter existing inputs. All plot implementations are emitted inline.
"""
from pathlib import Path
import textwrap
import nbformat as nbf

OUT=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s,stem=None):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip(),metadata={'output_stem':stem} if stem else {}))

md('''# ChronoClone Weinreb manuscript revision

## Scope
Rewrite the manuscript through Weinreb for the new Google Doc only.
The old Google Doc and old manuscript source are protected. No Singh analysis,
metaclone refitting, gene-program refitting, or package modification is performed.
At the authors' subsequent request, within-day clone-median omnibus tests are added
to the pseudotime panels, with a fixed joint multiple-testing family. Existing
gene-program contrasts receive visible significance annotations without refitting.
Existing figure-source tables support presentation changes, HSPC aggregation,
and direct displays of already calculated branch occupancies and distances.
The retained original pipeline and ODE illustrations are unchanged.

## Environment and execution
Environment **chronoclone**, existing Conda-compatible installation at
`/path/to/ChronoClone/.runtime/chronoclone`; kernel `chronoclone`.
No environment installation or mutation. Versions are printed below.
Execute with:
`/path/to/ChronoClone/.runtime/chronoclone/bin/python /path/to/ChronoClone/analysis_workspace/manuscript/chronoclone_panel_revision_20260910/execute_notebook.py /path/to/ChronoClone/analysis_workspace/manuscript/chronoclone_weinreb_v3_20260910/0001_manuscript_panels_and_evidence.ipynb`

This notebook is scaffolded by `build_revision_notebook.py`; its executed cells
are the reproducible entry point. Each plot-output cell includes its data
shaping, construction, editable SVG export, and display. Inherited images may
contain the original raster layers, which are not represented as new vectors.
Chart policy: existing MC palette; common cell-state palette with pooled HSPCs;
no narrative text or titles inside panels, only functional labels and keys.
''')
code('''
from pathlib import Path
import os, sys, json, hashlib, shutil, re, subprocess, platform
import importlib.metadata as im
ROOT=Path('/path/to/ChronoClone')
OUT=ROOT/'analysis_workspace/manuscript/chronoclone_weinreb_v3_20260910'
for name in ['figures','tables','reports','pre_edit_snapshot','runtime']:
    (OUT/name).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib')
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from IPython.display import display, SVG
ENV=Path('/path/to/ChronoClone/.runtime/chronoclone')
assert Path(sys.executable).resolve()==(ENV/'bin/python').resolve()
environment={'environment':'chronoclone','path':str(ENV),'manager':'Existing Conda-compatible environment; no mutation','kernel':'chronoclone','Python':platform.python_version(),**{p:im.version(p) for p in ['numpy','pandas','matplotlib','anndata','nbformat','scipy','statsmodels']}}
display(pd.DataFrame(environment.items(),columns=['field','value']))
(OUT/'reports/environment.json').write_text(json.dumps(environment,indent=2))
OLD=ROOT/'analysis_workspace/manuscript/chronoclone_panel_revision_20260910'
OLD_MS=ROOT/'analysis_workspace/manuscript_drafts/chronoclone_weinreb_manuscript_20260901/manuscript.md'
BENCH=ROOT/'analysis_workspace/benchmarks/chronoclone_method_noise_benchmark_day0_locked_20260812'
ANCESTRY=ROOT/'analysis_workspace/weinreb/weinreb_metaclone_ancestry_worked_examples_20260910'
MULTI=ROOT/'analysis_workspace/weinreb/weinreb_multilineage_metaclone_dynamics_20260901'
PROGRAM=ROOT/'analysis_workspace/weinreb/weinreb_day2_hspc_metaclone_gene_programs_20260902'
FIG=OUT/'figures'
NODE='~/.cache/research-runtimes/research-primary-runtime/dependencies/node/bin/node'
def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()
assert sha256(OLD_MS)=='23c7a92a3d4a78f65d3fbf0020beb258b65076c5f21c53cea51fba75165ca088'
if not (OUT/'pre_edit_snapshot/prior_manuscript.md').exists(): shutil.copy2(OLD_MS,OUT/'pre_edit_snapshot/prior_manuscript.md')
export=Path('/tmp/exports/ChronoClone v3-169d8fdf-7d23-4744-a6af-25acee8fd4a7.docx')
if export.exists() and not (OUT/'pre_edit_snapshot/new_google_doc_before.docx').exists(): shutil.copy2(export,OUT/'pre_edit_snapshot/new_google_doc_before.docx')
source_manifest=pd.read_csv(OLD/'reports/panel_manifest.csv')
for _,row in source_manifest.iterrows():
    assert sha256(ROOT/row.svg)==row.svg_sha256
    assert sha256(ROOT/row.png)==row.png_sha256
print('Registered source SVG/PNG assets verified:',len(source_manifest))
PALETTE=pd.read_csv(ROOT/'analysis_workspace/manuscript/chronoclone_manuscript_panel_styling_20260907/tables/manuscript_metaclone_palette.csv').set_index('label').color.to_dict()
INPUT=ROOT/'reference_workspace/data/klein_metaclone_palantir_trajectories.h5ad'
assert sha256(INPUT)=='4cb08c12dc70f5c3789a75a09a6399db06b3f6efe18147b9f2cca0908ba93911'
A=ad.read_h5ad(INPUT,backed='r')
obs=A.obs.copy()
coordinates={key:np.asarray(A.obsm[key]) for key in A.obsm.keys()}
state_order=list(obs.cell_type.cat.categories)
STATE_COLORS=dict(zip(state_order,A.uns['cell_type_colors']))
A.file.close()
print('Coordinates:',{k:v.shape for k,v in coordinates.items()})
display(pd.DataFrame({'state':state_order,'color':[STATE_COLORS[s] for s in state_order]}))
print('Observed days:',obs.day.value_counts().to_dict())
NS='{http://www.w3.org/2000/svg}'
ET.register_namespace('',NS[1:-1]);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
mpl.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
panel_records=[]
''')
md('## Existing quantitative evidence')
code('''
effects=pd.read_csv(ANCESTRY/'tables/evidence_review_selected_examples.csv')
display(effects)
occupancy=pd.read_csv(ANCESTRY/'tables/ancestry_03_multilineage_branch_occupancy_change.csv')
pair_distances=pd.read_csv(ANCESTRY/'tables/ancestry_multilineage_branch_pair_distances.csv')
display(occupancy)
display(pair_distances[(pair_distances.cohort=='day2_hspc_linked')&(pair_distances.estimator=='clone_balanced_occupancy')])
config=json.loads((BENCH/'config/benchmark_config.json').read_text())
print('Benchmark display time labels:', ['t1','t2','t3','t4'])
print('Registered method parameters:', {k:config[k] for k in ['repeats','master_seed','min_persistence','min_total_count','chronoclone','clades','generic_baselines']})
print('Benchmark summary columns:',pd.read_csv(BENCH/'results/aggregate/native_method_noise_summary.csv',nrows=1).columns.tolist())
''')

# Additional dedicated plot-output cells are appended below during authoring.
extra=OUT/'revision_plot_cells.py'
if extra.exists():
    scope={'md':md,'code':code}
    exec(compile(extra.read_text(),str(extra),'exec'),scope)

notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'chronoclone','language':'python','name':'chronoclone'},'language_info':{'name':'python'}})
nbf.write(notebook,OUT/'0001_manuscript_panels_and_evidence.ipynb')
print(OUT/'0001_manuscript_panels_and_evidence.ipynb')
