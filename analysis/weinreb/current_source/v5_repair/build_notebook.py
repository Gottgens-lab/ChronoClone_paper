from pathlib import Path
import nbformat

OUT = Path(__file__).resolve().parent
cells = []
def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
def code(text, stem=None):
    c = nbformat.v4.new_code_cell(text)
    if stem: c.metadata.update({'tags':['plot-cell'], 'output_stem':stem})
    cells.append(c)

md('''# Branch-restricted Weinreb pseudotime panels

## Scope and environment
Repair only Figure 2E and Figure 2G so that displayed cell distributions and existing branch-level clone summaries use the same cohort. Existing metaclones, Palantir coordinates, branch masks and 12-test omnibus results remain unchanged. No new hypothesis tests or model fits.

Existing manager/name: conda / **chronoclone**. MacBook-local path: `/path/to/ChronoClone/.runtime/chronoclone`; kernel: `chronoclone`; Python 3.11.15. Exact installed versions are printed below.

Execution command: `/path/to/ChronoClone/.runtime/chronoclone/bin/python /path/to/ChronoClone/analysis_workspace/manuscript/chronoclone_academic_revision_20260911/figure_repairs/execute_notebook.py /path/to/ChronoClone/analysis_workspace/manuscript/chronoclone_academic_revision_20260911/figure_repairs/0001_branch_restricted_pseudotime.ipynb`

Figures retain the original categorical palette, global pseudotime orientation, grouped violin style and black cell-median bars. Groups with fewer than eight cells show each cell as an open point and a cell-count label without a fitted density. Stars are copied from the saved q12 family. They indicate within-day omnibus differences, not every pairwise contrast; unsupported groups remain visible but do not enter their day's test.
''')
code('''from pathlib import Path
import json, hashlib, sys, importlib.metadata as im
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from IPython.display import display
ROOT=Path('/path/to/ChronoClone')
OUT=ROOT/'analysis_workspace/manuscript/chronoclone_academic_revision_20260911/figure_repairs'
PANELS=OUT/'panels'; TABLES=OUT/'tables'; REPORTS=OUT/'reports'
for folder in [PANELS,TABLES,REPORTS]: folder.mkdir(parents=True,exist_ok=True)
environment={'manager':'conda (existing; not modified)','name':'chronoclone','path':'/path/to/ChronoClone/.runtime/chronoclone','kernel':'chronoclone','python':sys.version.split()[0],'executable':sys.executable,'versions':{p:im.version(p) for p in ['anndata','numpy','pandas','matplotlib','nbformat','nbclient']}}
(REPORTS/'environment.json').write_text(json.dumps(environment,indent=2))
display(environment)
mpl.rcParams.update({'font.family':'Arial','font.size':12.825,'axes.edgecolor':'#4B5563','axes.labelcolor':'#172033','axes.linewidth':.8,'xtick.color':'#4B5563','ytick.color':'#4B5563','text.color':'#172033','svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
''')
md('## Locked inputs and cohort')
code('''SOURCE=ROOT/'reference_workspace/data/klein_metaclone_palantir_trajectories.h5ad'
OLD=ROOT/'analysis_workspace/manuscript/chronoclone_grid_revision_20260911'
PALETTE=ROOT/'analysis_workspace/manuscript/chronoclone_manuscript_panel_styling_20260907/tables/manuscript_metaclone_palette.csv'
TESTS=OLD/'supplementary_tables/Table_S2/pseudotime_within_day_omnibus_tests.csv'
MEDIANS=OLD/'supplementary_tables/Table_S2/pseudotime_omnibus_clone_medians.csv'
source_files=[SOURCE,PALETTE,TESTS,MEDIANS]
source_manifest=pd.DataFrame([{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_files])
assert source_manifest.iloc[0].sha256=='4cb08c12dc70f5c3789a75a09a6399db06b3f6efe18147b9f2cca0908ba93911'
source_manifest.to_csv(REPORTS/'source_manifest.csv',index=False)
a=ad.read_h5ad(SOURCE,backed='r')
obs=a.obs.drop(columns=['population'],errors='ignore').copy()
obs['metaclone']=pd.to_numeric(obs.metaclone,errors='coerce');obs['day']=pd.to_numeric(obs.day,errors='coerce')
obs['cell']=obs.index.astype(str);obs['LARRY_ID']=obs.LARRY_ID.astype(str)
hspc=obs[(obs.day==2)&(obs.metaclone>=0)&obs.cell_type.isin(['prog_1','prog_2'])]
linked=set(hspc.LARRY_ID)
assert len(linked)==569 and len(hspc)==775
obs=obs[(obs.metaclone>=0)&obs.LARRY_ID.isin(linked)].copy()
assert len(obs)==7733 and obs.metaclone.nunique()==20
colors=pd.read_csv(PALETTE).set_index('metaclone').color.to_dict()
tests=pd.read_csv(TESTS);saved_medians=pd.read_csv(MEDIANS)
assert len(tests)==12 and tests.test_family.nunique()==1
tests.to_csv(TABLES/'unchanged_q12_tests.csv',index=False)
display(source_manifest)
display(pd.DataFrame({'cohort':['Day-2 HSPCs','Assigned linked cells'],'cells':[len(hspc),len(obs)],'clones':[len(linked),obs.LARRY_ID.nunique()]}))
checks=[]
''')

template='''STEM=__STEM__; BRANCH=__BRANCH__; CONTEXT=__CONTEXT__; MCS=__MCS__; DAYS=[2,4,6]
# Plot-specific data shaping, saved-summary equivalence and figure construction.
plot_data=obs[obs.metaclone.isin(MCS)&obs.day.isin(DAYS)&obs['mask_'+BRANCH].astype(bool)].copy()
plot_data['pseudotime']=pd.to_numeric(plot_data.palantir_pseudotime)
plot_data=plot_data[np.isfinite(plot_data.pseudotime)]
plot_data=plot_data[['cell','LARRY_ID','metaclone','day','pseudotime']].sort_values(['day','metaclone','cell'])
assert plot_data.cell.is_unique and plot_data.LARRY_ID.isin(linked).all()
clone_medians=plot_data.groupby(['metaclone','day','LARRY_ID'],observed=True).pseudotime.median().reset_index()
comparison=clone_medians.merge(saved_medians[saved_medians.context==CONTEXT],on=['metaclone','day','LARRY_ID'],how='outer',suffixes=('_current','_saved'),indicator=True,validate='one_to_one')
assert comparison['_merge'].eq('both').all()
assert np.allclose(comparison.pseudotime_current,comparison.pseudotime_saved,rtol=0,atol=1e-14)
summary=plot_data.groupby(['metaclone','day'],observed=True).agg(n_cells=('cell','size'),n_clones=('LARRY_ID','nunique'),cell_median=('pseudotime','median')).reset_index()
summary['branch']=BRANCH
selected_tests=tests[tests.context==CONTEXT].set_index('day').loc[DAYS]
for day,row in selected_tests.iterrows():
    expected={int(k):v for k,v in json.loads(row.n_clones_by_metaclone).items()}
    actual=clone_medians[clone_medians.day==day].groupby('metaclone').size().to_dict()
    assert actual==expected
plot_data.to_csv(TABLES/(STEM+'_cells.csv'),index=False)
clone_medians.to_csv(TABLES/(STEM+'_clone_medians.csv'),index=False)
summary.to_csv(TABLES/(STEM+'_summary.csv'),index=False)
comparison.to_csv(TABLES/(STEM+'_saved_median_checks.csv'),index=False)
figure_count_before=len(plt.get_fignums())
fig,ax=plt.subplots(figsize=(7.34,4.21))
fig.subplots_adjust(left=.135,right=.985,bottom=.13,top=.88)
width=.80/len(MCS)
for day_index,day in enumerate(DAYS):
    for mc_index,mc in enumerate(MCS):
        values=plot_data.loc[(plot_data.day==day)&(plot_data.metaclone==mc),'pseudotime'].to_numpy(float)
        x=day_index+(mc_index-(len(MCS)-1)/2)*width
        if len(values)>=8 and np.unique(values).size>=2:
            violin=ax.violinplot([values],positions=[x],widths=width*.92,showmedians=True,showextrema=False)
            for body in violin['bodies']:
                body.set_facecolor(colors[mc]);body.set_edgecolor(colors[mc]);body.set_linewidth(.65);body.set_alpha(.86)
            violin['cmedians'].set_color('#111827');violin['cmedians'].set_linewidth(1.1)
        elif len(values):
            # Deterministic horizontal offsets reveal every sparse observation.
            point_x=np.linspace(x-width*.23,x+width*.23,len(values)) if len(values)>1 else np.array([x])
            ax.scatter(point_x,values,s=14,facecolors='none',edgecolors=colors[mc],linewidths=1.0,zorder=3)
            ax.plot([x-width*.23,x+width*.23],[np.median(values)]*2,color='#111827',lw=1.1)
            ax.text(x,float(np.max(values))+.025,f'n={len(values)}',ha='center',va='bottom',fontsize=10,color=colors[mc])
    ax.text(day_index,1.075,selected_tests.loc[day,'symbol'],ha='center',va='bottom',fontsize=18,fontweight='bold',color='#111111')
ax.set_xticks(range(3),[f'Day {day}' for day in DAYS],fontsize=12.825)
ax.set_ylabel('Palantir pseudotime',fontsize=12.825)
ax.set_ylim(0,1.17);ax.set_xlim(-.50,2.50)
ax.set_yticks(np.arange(0,1.01,.2));ax.tick_params(axis='y',labelsize=11.475)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',color='#E5E7EB',linewidth=.6);ax.set_axisbelow(True)
assert len(plt.get_fignums())==figure_count_before+1 and not ax.get_title()
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
text_artists=[ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]
for artist in text_artists:
    box=artist.get_window_extent(renderer)
    assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
fig.savefig(PANELS/(STEM+'.svg'),bbox_inches='tight',pad_inches=.06)
fig.savefig(PANELS/(STEM+'.png'),dpi=400,bbox_inches='tight',pad_inches=.06)
display(fig);plt.close(fig)
display(summary)
checks.append({'stem':STEM,'branch':BRANCH,'cells':len(plot_data),'clone_day_records':len(clone_medians),'saved_clone_medians_max_error':float(np.max(np.abs(comparison.pseudotime_current-comparison.pseudotime_saved))),'saved_test_membership_identical':True,'new_hypothesis_tests':0,'figure_count':1,'text_inside_frame':True,'svg_sha256':hashlib.sha256((PANELS/(STEM+'.svg')).read_bytes()).hexdigest(),'png_sha256':hashlib.sha256((PANELS/(STEM+'.png')).read_bytes()).hexdigest()})
print(PANELS/(STEM+'.svg'))
'''
for stem,branch,context,mcs in [('figure_2e_neutrophil_pseudotime','Neu','neutrophil_output',[1,22,9,16,14]),('figure_2g_monocyte_cell_pseudotime','Mono','monocyte_output',[6,8,7,12])]:
    md('## '+('Figure 2E' if branch=='Neu' else 'Figure 2G'))
    source=template.replace('__STEM__',repr(stem)).replace('__BRANCH__',repr(branch)).replace('__CONTEXT__',repr(context)).replace('__MCS__',repr(mcs))
    code(source,stem)
md('## Verification receipt')
code('''check_table=pd.DataFrame(checks)
assert len(check_table)==2 and check_table.saved_test_membership_identical.all()
assert len(list(PANELS.glob('*.svg')))==2
check_table.to_csv(REPORTS/'panel_checks.csv',index=False)
display(check_table)
print('Only the two displayed cell cohorts were repaired. Saved clone medians and omnibus test membership agree; original data, masks and test results are unchanged.')
''')
book=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'chronoclone','display_name':'chronoclone','language':'python'},'language_info':{'name':'python'}})
nbformat.validate(book)
nbformat.write(book,OUT/'0001_branch_restricted_pseudotime.ipynb')
