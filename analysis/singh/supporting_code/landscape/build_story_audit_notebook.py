"""Build a table-only audit of already computed, corrected Singh reference profiles."""
from pathlib import Path
import nbformat as nb

O=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))
md('''# Descriptive audit of Singh reference histories

## Context and scope
This table-only companion supports a read-only manuscript storyline review. It re-summarizes existing corrected family/sample rows, without expression access, reclustering, hypothesis tests, new effect tests, or modification of the source checkpoints. Tables cover every frozen WT-reference metaclone and the selected MC0, MC3 and MC7 examples. No figures are created because the parent task owns visual production.

Within each metaclone, day and sequencing sample, cells have already been averaged within lineage family in the corrected source. For a descriptive WT profile, family rows are equally weighted within each observed sample, then sample means are equally weighted. These weights match the current WT-reference pseudotime panel and descriptive MC3 composition panel, but are not the fixed minimum-support contrast weights used for inference. Profiles at different times can comprise different observed families; the summaries are not individual-cell trajectories or growth rates. Day 7 is the single shared pre-induction WT population. Missing Day-27 WT recovery is not extinction or zero composition. The deposited reference partition remains fixed.

Environment: **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`; kernel **chronoclone**. Versions, executable, exact execution command and input hashes are printed below. No packages are installed.
''')
code(r'''
from pathlib import Path
import sys,json,hashlib,platform,importlib.metadata as metadata
import numpy as np,pandas as pd
from IPython.display import display
ROOT=Path('/path/to/ChronoClone')
O=ROOT/'analysis_workspace/singh/singh_landscape_expansion_20260911'
SRC=ROOT/'analysis_workspace/singh/singh_reference_response_20260911/dynamics'
for directory in ['tables','reports']: (O/directory).mkdir(parents=True,exist_ok=True)
environment={'environment':'chronoclone','path':str(Path(sys.executable).parent.parent),'kernel':'chronoclone','Python':platform.python_version(),'versions':{p:metadata.version(p) for p in ['numpy','pandas','nbformat','nbclient']},'execution_command':f'{sys.executable} {ROOT/"analysis_workspace/manuscript/chronoclone_panel_revision_20260910/execute_notebook.py"} {O/"0001_reference_story_audit.ipynb"}'}
assert environment['path']=='/path/to/ChronoClone/.runtime/chronoclone'
display(environment)
inputs=[SRC/'tables/clone_arm_table.csv',SRC/'tables/cells_resolved.csv',SRC/'tables/wt_reference_position.csv',SRC/'tables/support_by_metaclone.csv',SRC/'tables/effects.csv',SRC/'tables/mc3_day27_arm_composition.csv',SRC.parent/'build_core_notebook.py',ROOT/'analysis_workspace/manuscript/chronoclone_v4_20260911/manuscript_source.md']
manifest=pd.DataFrame([{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in inputs])
manifest.to_csv(O/'reports/story_input_manifest.csv',index=False)
display(manifest)
CA=pd.read_csv(SRC/'tables/clone_arm_table.csv',dtype={'mc':str,'clone':str})
D=pd.read_csv(SRC/'tables/cells_resolved.csv',dtype={'mc':str,'clone':str})
S=pd.read_csv(SRC/'tables/support_by_metaclone.csv',dtype={'mc':str})
E=pd.read_csv(SRC/'tables/effects.csv',dtype={'mc':str})
CELLTYPES=['HSPC','GMP','Mk','Ery','Basophil','Monocyte','Neutrophil']
MCS=sorted(D.mc.unique(),key=int);DAYS=['D7','D14','D27']
assert len(MCS)==17 and not CA.duplicated(['mc','day','arm','stratum','clone']).any()
assert D.cell_id.is_unique and D.groupby('clone').mc.nunique().max()==1
assert D.loc[D.day.eq('D7'),'arm'].eq('WT').all()
assert np.allclose(CA[CELLTYPES].sum(axis=1),1)
check_support=D.groupby(['mc','day','arm'],observed=True).agg(cells=('clone','size'),families=('clone','nunique')).reset_index()
joined=S.merge(check_support,on=['mc','day','arm'],suffixes=('_saved','_checked'),how='outer',validate='one_to_one')
assert joined[['cells_saved','cells_checked','families_saved','families_checked']].notna().all().all()
assert joined.cells_saved.eq(joined.cells_checked).all() and joined.families_saved.eq(joined.families_checked).all()
assert int(CA.n.sum())==len(D)==14896 and D.clone.nunique()==1516
print('Corrected source grain, unique cells, one metaclone per family, fractions, Day-7 arm, and complete support registry verified.')
''')
md('## WT reference profiles for all metaclones\nThe full table keeps all planned WT days. Missing profiles remain NaN; zero support is displayed separately. Top output categories are descriptive summaries, not newly assigned metaclone identities.')
code(r'''
sample_profiles=CA.loc[CA.arm.eq('WT')].groupby(['mc','day','stratum'],observed=True)[['psi']+CELLTYPES].mean()
profile=sample_profiles.groupby(['mc','day'],observed=True).mean()
index=pd.MultiIndex.from_product([MCS,DAYS],names=['mc','day'])
profile=profile.reindex(index)
samples=sample_profiles.reset_index().groupby(['mc','day']).stratum.nunique().reindex(index,fill_value=0)
wt_support=S.loc[S.arm.eq('WT')].set_index(['mc','day'])[['cells','families']].reindex(index,fill_value=0)
profile=profile.join(wt_support).join(samples.rename('observed_samples'))
saved=pd.read_csv(SRC/'tables/wt_reference_position.csv',dtype={'mc':str}).set_index('mc').reindex(MCS)
assert np.allclose(profile.psi.unstack('day').reindex(index=MCS,columns=DAYS),saved[DAYS],equal_nan=True,atol=1e-12)
present=profile.cells.gt(0)
assert np.allclose(profile.loc[present,CELLTYPES].sum(axis=1),1)
assert profile.loc[~present,['psi']+CELLTYPES].isna().all().all()
assert profile.loc[profile.index.get_level_values('day')=='D7','HSPC'].eq(1).all()
profile['top_three']=profile.apply(lambda row:'; '.join(f'{label} {100*value:.1f}%' for label,value in row[CELLTYPES].dropna().nlargest(3).items() if value>0) if row.cells else 'Not recovered',axis=1)
profile.reset_index().to_csv(O/'tables/wt_reference_profiles.csv',index=False)
sample_profiles.reset_index().to_csv(O/'tables/wt_sample_profiles.csv',index=False)
wide=[]
for mc in MCS:
    row={'metaclone':'MC'+mc}
    for day in DAYS:
        r=profile.loc[(mc,day)]
        row[day+' pseudotime']=round(r.psi,3) if r.cells else np.nan
        row[day+' cells/families']=f'{int(r.cells)}/{int(r.families)}'
        if day!='D7':row[day+' composition']=r.top_three
    wide.append(row)
wide=pd.DataFrame(wide)
wide.to_csv(O/'tables/wt_reference_story_registry.csv',index=False)
display(wide)
print('Every non-missing WT pseudotime profile reproduces the existing reference panel to 1e-12.')
''')
md('## Representative references and mutant observations\nMC0, MC3 and MC7 are narrative examples selected in the existing manuscript, not a new screening result. Their descriptive profiles are accompanied by observed family counts. The mutation-response tests are only read from the existing corrected results; none are recalculated or added.')
code(r'''
selected=CA.loc[CA.mc.isin(['0','3','7'])]
sample_selected=selected.groupby(['mc','day','arm','stratum'],observed=True)[['psi']+CELLTYPES].mean()
selected_profile=sample_selected.groupby(['mc','day','arm'],observed=True).mean().reset_index()
selected_profile=selected_profile.merge(S,on=['mc','day','arm'],how='left',validate='one_to_one')
selected_profile['top_three']=selected_profile.apply(lambda row:'; '.join(f'{label} {100*value:.1f}%' for label,value in row[CELLTYPES].astype(float).nlargest(3).items() if value>0),axis=1)
selected_profile.to_csv(O/'tables/representative_observed_profiles.csv',index=False)
saved_composition=pd.read_csv(SRC/'tables/mc3_day27_arm_composition.csv').set_index('arm').reindex(['WT','Dnmt3a','Npm1c'])
reproduced=selected_profile.loc[selected_profile.mc.eq('3')&selected_profile.day.eq('D27')].set_index('arm').reindex(['WT','Dnmt3a','Npm1c'])
assert np.allclose(reproduced[CELLTYPES],saved_composition[CELLTYPES],atol=1e-12)
display(selected_profile[['mc','day','arm','psi','cells','families','top_three']])
selected_effects=E.loc[E.mc.isin(['0','3','7']),['mc','day','mutant','point_estimable','inference_estimable','dpsi','ci_low','ci_high','q_state','tvd','q_composition','reason']]
selected_effects.to_csv(O/'tables/representative_existing_effects.csv',index=False)
display(selected_effects)
print('MC3 Day-27 descriptive compositions reproduce the existing panel to 1e-12. Existing tests copied without modification.')
''')
md('## Boundaries and reproducibility\nNo differential, enrichment, clustering, branch, trajectory or mutation-response test is introduced. Observed profile labels do not establish self-renewal, exhaustion, rescue, reversal, fate determinism or independently replicated mutation effects. The submitted story review identifies where the current manuscript can be expanded using these descriptive quantities.')
code(r'''
products=sorted((O/'tables').glob('*.csv'))+[O/'reports/story_input_manifest.csv']
output_manifest=pd.DataFrame([{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in products])
output_manifest.to_csv(O/'reports/story_output_manifest.csv',index=False)
display(output_manifest)
print('Completed table-only descriptive audit; no new statistical tests and no source checkpoint changes.')
''')
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (chronoclone)','language':'python','name':'chronoclone'},'language_info':{'name':'python','version':'3.11.15'}})
nb.validate(notebook)
nb.write(notebook,O/'0001_reference_story_audit.ipynb')
