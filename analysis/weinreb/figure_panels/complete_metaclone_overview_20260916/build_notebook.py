from pathlib import Path
import nbformat as nb
R=Path('/path/to/ChronoClone');O=R/'analysis_workspace/weinreb/20260916_complete_metaclone_overview'
v3=nb.read(R/'ChronoClone_paper/analysis/weinreb/current_source/v3_display/0001_manuscript_panels_and_evidence.ipynb',as_version=4)
old=nb.read(R/'analysis_workspace/weinreb/weinreb_multilineage_metaclone_dynamics_20260901/0001_weinreb_multilineage_metaclone_dynamics.ipynb',as_version=4)
c=[]
def md(s):c.append(nb.v4.new_markdown_cell(s))
def code(s):c.append(nb.v4.new_code_cell(s))
md('''# Complete Weinreb metaclone overview before analysis

The authors requested identical sizes for all time-resolved metaclone UMAPs, a complete overview before analysis, and no silent omission of MC4/MC11. The five family grids below cover all retained metaclones at Days 2/4/6. Every UMAP uses the same axes box, coordinate limits, aspect, marker geometry and typography. Grids have equal height; insert them at equal heights, not equal widths. The seven multilineage metaclones shown in their overview define the downstream analysis cohort directly.

The original dynamics notebook explicitly described an earlier requested five-metaclone comparison; its hard-coded list propagated to manuscript composition and pseudotime panels. The later atlas showed seven. This revision resolves that mismatch, without retrospectively claiming that the prior five-member analysis covered all seven.

Existing micromamba **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`, kernel **chronoclone-paper**. Exact versions print below. Execute:
```sh
JUPYTER_PATH=/path/to/ChronoClone/.runtime/jupyter /path/to/ChronoClone/.runtime/chronoclone/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=chronoclone-paper /path/to/ChronoClone/analysis_workspace/weinreb/20260916_complete_metaclone_overview/complete_overview_and_analysis.ipynb
```

Density keeps the author's assigned-cell reference, 100 nearest neighbors and within-metaclone color maximum across days. It is not a between-metaclone abundance scale. Statistical comparisons use clone medians, retain the established twelve-test BH family, and explicitly report support. Branch restriction defines the estimand; low-support groups remain displayed. No expression analysis or new embedding is run. Historical results are preserved.
''')
code('''from pathlib import Path
import os,sys,json,hashlib,platform,importlib.metadata as im
ROOT=Path('/path/to/ChronoClone')
OUT=ROOT/'analysis_workspace/weinreb/20260916_complete_metaclone_overview'
FIG=OUT/'figures';TABLE_DIR=OUT/'tables';FIGURE_DIR=FIG
for p in [FIG,TABLE_DIR,OUT/'reports']:p.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import numpy as np,pandas as pd,anndata as ad,matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from IPython.display import display
from PIL import Image
ENV=ROOT/'.runtime/chronoclone'
assert Path(sys.executable).resolve()==(ENV/'bin/python').resolve()
print({'environment':'chronoclone','path':str(ENV),'kernel':'chronoclone-paper','Python':platform.python_version(),**{p:im.version(p) for p in ['anndata','numpy','pandas','matplotlib','scipy','scikit-learn','statsmodels']}})
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCE=ROOT/'reference_workspace/data/klein_metaclone_palantir_trajectories.h5ad'
assert sha256(SOURCE)=='4cb08c12dc70f5c3789a75a09a6399db06b3f6efe18147b9f2cca0908ba93911'
a=ad.read_h5ad(SOURCE,backed='r');obs=a.obs.copy();U=np.asarray(a.obsm['X_umap']).copy();state_order=list(obs.cell_type.cat.categories);a.file.close()
cell_type_order=state_order
required_columns=['day','cell_type','cell_type_weinreb','LARRY_ID','metaclone','palantir_pseudotime','mask_Baso','mask_Neu','mask_Mono']
obs['metaclone']=pd.to_numeric(obs.metaclone);obs['day']=pd.to_numeric(obs.day)
GROUPS={'neutrophil':[9,14,16,22,1],'monocyte':[12,7,8,6],'erythroid_mk':[10,17,23],'basophil':[13],'multilineage':[0,2,5,3,18,4,11]}
ALL_MCS=[m for group in GROUPS.values() for m in group];DAYS=[2,4,6]
assert len(ALL_MCS)==len(set(ALL_MCS))==20
assert set(ALL_MCS)==set(obs.loc[obs.metaclone.ge(0),'metaclone'].astype(int))
assigned=obs.metaclone.ge(0).to_numpy();full_umap=U[assigned];mc_vals=obs.loc[assigned,'metaclone'].to_numpy();day_vals=obs.loc[assigned,'day'].to_numpy()
nbrs=NearestNeighbors(n_neighbors=100,algorithm='ball_tree').fit(full_umap)
PLOT_XLIM=(U[:,0].min()-1,U[:,0].max()+1);PLOT_YLIM=(U[:,1].min()-1,U[:,1].max()+1)
PALETTE=pd.read_csv(ROOT/'ChronoClone_paper/analysis/weinreb/inputs/manuscript_metaclone_palette.csv').set_index('label').color.to_dict()
MC_COLORS={m:PALETTE[f'MC{m}'] for m in ALL_MCS}
mpl.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':13,'figure.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
geometry=[];density_support=[];panel_records=[]
support=obs[obs.metaclone.isin(ALL_MCS)].groupby(['metaclone','day'],observed=True).agg(cells=('LARRY_ID','size'),clones=('LARRY_ID','nunique')).reindex(pd.MultiIndex.from_product([ALL_MCS,DAYS],names=['metaclone','day']),fill_value=0).reset_index()
support.to_csv(TABLE_DIR/'all_metaclone_day_support.csv',index=False)
display(support[support.metaclone.isin([4,11])])
''')
for fam in ['neutrophil','monocyte','erythroid_mk','basophil','multilineage']:
 md('## Overview: '+fam.replace('_',' '))
 code(f"family={fam!r}\n"+'''metaclones=GROUPS[family];stem='overview_'+family
# Plot-specific density calculation and scaling match the original author code.
cache={};vmax={}
for mc in metaclones:
    for day in DAYS:
        idx=np.flatnonzero((mc_vals==mc)&(day_vals==day))
        if len(idx):
            neighbors=nbrs.kneighbors(full_umap[idx],return_distance=False)
            dens=np.isin(neighbors,idx).sum(axis=1)
        else:dens=np.array([],dtype=int)
        cache[mc,day]=(idx,dens)
    vmax[mc]=max(5,max((d.max() if len(d) else 0) for (_,d) in [cache[mc,day] for day in DAYS]))
# Fixed physical axes and row/column pitch; column count changes canvas width only.
fw=1.05+1.55*len(metaclones)+.95;fh=6.15
fig=plt.figure(figsize=(fw,fh),dpi=160)
labels=[]
for j,mc in enumerate(metaclones):
    labels.append(fig.text((1.05+1.55*(j+.5))/fw,1-.10/fh,f'MC{mc}',ha='center',va='top',fontsize=14,fontweight='bold',color=MC_COLORS[mc]))
    for i,day in enumerate(DAYS):
        bottom=fh-.45-1.9*(i+1)+.26
        ax=fig.add_axes([(1.05+1.55*j)/fw,bottom/fh,1.52/fw,1.66/fh])
        ax.set_xlim(PLOT_XLIM);ax.set_ylim(PLOT_YLIM);ax.set_aspect('equal');ax.set_axis_off()
        ax.scatter(U[:,0],U[:,1],s=3.5,c='lightgrey',linewidths=0,rasterized=True,zorder=0)
        idx,dens=cache[mc,day];order=np.argsort(dens,kind='stable')
        if len(idx):ax.scatter(full_umap[idx[order],0],full_umap[idx[order],1],c=dens[order],cmap='viridis',vmin=0,vmax=vmax[mc],s=50,alpha=.8,edgecolors='black',linewidths=.15,rasterized=False,zorder=2)
        ncl=obs[(obs.metaclone==mc)&(obs.day==day)].LARRY_ID.nunique()
        labels.append(fig.text((1.05+1.55*(j+.5))/fw,(bottom-.035)/fh,f'n={len(idx)}  C={ncl}',ha='center',va='top',fontsize=11,color='#626972'))
        if j==0:labels.append(fig.text(.88/fw,(bottom+.83)/fh,f'Day {day}',ha='right',va='center',fontsize=13,fontweight='bold'))
        fig.canvas.draw();box=ax.get_window_extent()
        geometry.append({'stem':stem,'mc':mc,'day':day,'width_pt':box.width/fig.dpi*72,'height_pt':box.height/fig.dpi*72})
        density_support.append({'mc':mc,'day':day,'cells':len(idx),'clones':ncl,'density_vmax':int(vmax[mc])})
bar=fig.add_axes([(1.05+1.55*len(metaclones)+.15)/fw,.32,.11/fw,1.5/fh])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.Normalize(0,1),cmap='viridis'),cax=bar)
cb.set_ticks([0,1],labels=['Low','High']);cb.ax.tick_params(labelsize=10,length=0);cb.outline.set_visible(False)
labels.append(fig.text((1.05+1.55*len(metaclones)+.15)/fw,.32+1.5/fh+.025,'Relative\\ndensity',fontsize=11,va='bottom'))
fig.canvas.draw()
for label in labels:
    box=label.get_window_extent(fig.canvas.get_renderer());assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height,(stem,label.get_text())
fig.savefig(FIG/f'{stem}.svg');fig.savefig(FIG/f'{stem}.png',dpi=300)
display(fig);plt.close(fig)
panel_records.append({'stem':stem,'metaclones':metaclones,'svg_sha256':sha256(FIG/f'{stem}.svg'),'png_sha256':sha256(FIG/f'{stem}.png')})
print(FIG/f'{stem}.png')
''')
md('## Analysis cohort and display palette\nThe complete multilineage overview above defines the analysis membership. No additional metaclone selection occurs below.')
s=old.cells[4].source.replace('TARGET_MCS = [0, 2, 5, 3, 18]',"TARGET_MCS = GROUPS['multilineage'].copy()")
code(s)
code(v3.cells[71].source)
# Reuse the original clone-balanced estimator verbatim, now including all seven.
code(old.cells[5].source)
# Composition chart: updated cells from v3 with expanded cohort; preserve maturity ordering.
s=v3.cells[81].source
start=s.index("snapshot=OUT/");end=s.index('plot_data=',start)
s=s[:start]+"original=clone_balanced_composition.copy()\n"+s[end:]
s=s.replace("source=MULTI/'tables/fine_state_composition_clone_balanced.csv'","source=SOURCE")
s=s.replace('[0,2,5,3,18]','TARGET_MCS').replace('figsize=(9.2,8.0)','figsize=(9.2,10.4)')
code(s)
# Recalculate established within-day family, including all displayed multilineage MCs.
tests=next(x.source for x in v3.cells if x.cell_type=='code' and 'PSEUDOTIME_TEST_DESIGNS={' in x.source)
tests=tests.replace("'mcs':[0,2,5,3,18]","'mcs':TARGET_MCS")
a=tests.index('legacy=pd.read_csv');b=tests.index('pd.concat(test_data',a);tests=tests[:a]+tests[b:]
tests=tests.replace("print('Saved historical multilineage H and P values unchanged; displayed q values use the declared joint 12-test family.')","print('Recomputed full-cohort tests; historical five-MC files preserved. These q values supersede the old restricted-family annotations.')")
code(tests)
for idx,branch in [(10,'Baso'),(11,'Neu'),(12,'Mono')]:
 md('## '+branch+' branch pseudotime: all seven multilineage metaclones')
 s=old.cells[idx].source.replace('figsize=(7.2, 4.35)','figsize=(9.2, 4.8)').replace('width = 0.15','width = 0.11').replace('(mc_position - 2)','(mc_position - (len(TARGET_MCS)-1)/2)')
 # Add visible support for every cohort member and recomputed q annotation.
 anchor='ax.set_xticks([0, 1])'
 extra='''for dp,day in enumerate(TARGET_DAYS):
    for mp,mc in enumerate(TARGET_MCS):
        count=len(branch_clone_medians[(branch_clone_medians.day==day)&(branch_clone_medians.metaclone==mc)])
        ax.text(dp+(mp-(len(TARGET_MCS)-1)/2)*width,1.035,f'C={count}',ha='center',fontsize=8,color=MC_COLORS[mc],rotation=90,va='bottom')
    test=pseudotime_tests[(pseudotime_tests.context=='multilineage_'+branch)&(pseudotime_tests.day==day)].iloc[0]
    ax.text(dp,1.19,f"q={test.q_value:.3g}" if test.estimable else 'Not estimable',ha='center',fontsize=11)
branch_clone_medians.to_csv(TABLE_DIR/f'{branch.lower()}_clone_medians_complete.csv',index=False)
'''
 s=s.replace(anchor,extra+'\n'+anchor).replace('ax.set_ylim(0, 1.04)','ax.set_ylim(0, 1.25);ax.set_yticks(np.linspace(0,1,6))')
 s=s.replace('ax.set_ylabel("Palantir pseudotime", fontsize=11.2)',f'ax.set_ylabel("Palantir pseudotime ({branch}-compatible cells)", fontsize=11.2)')
 s=s.replace('plt.show()',"display(fig);plt.close(fig)\npanel_records.append({'stem':branch_stem.name,'metaclones':TARGET_MCS,'svg_sha256':sha256(branch_stem.with_suffix('.svg')),'png_sha256':sha256(branch_stem.with_suffix('.png'))})")
 code(s)
md('## Completeness and geometry checks')
code('''g=pd.DataFrame(geometry);d=pd.DataFrame(density_support)
assert len(g)==60 and not g[['mc','day']].duplicated().any()
assert set(g.mc)==set(ALL_MCS)
assert np.allclose(g.width_pt,g.width_pt.iloc[0],rtol=1e-12) and np.allclose(g.height_pt,g.height_pt.iloc[0],rtol=1e-12)
assert set(TARGET_MCS)==set(GROUPS['multilineage']) and {4,11}.issubset(TARGET_MCS)
assert set(clone_balanced_composition.metaclone)==set(TARGET_MCS)
# Original five groups retain exactly the same clone-balanced composition.
legacy=pd.read_csv(ROOT/'analysis_workspace/weinreb/weinreb_multilineage_metaclone_dynamics_20260901/tables/fine_state_composition_clone_balanced.csv')
current=clone_balanced_composition[clone_balanced_composition.metaclone.isin([0,2,5,3,18])]
keys=['metaclone','day','cell_type']
audit=legacy.merge(current,on=keys,suffixes=('_old','_new'),validate='one_to_one')
assert len(audit)==len(legacy) and np.allclose(audit.fraction_old,audit.fraction_new,rtol=1e-12,atol=1e-14)
g.to_csv(TABLE_DIR/'umap_dimensions.csv',index=False);d.to_csv(TABLE_DIR/'density_support.csv',index=False)
qc={'retained_metaclones':ALL_MCS,'umap_count':len(g),'umap_width_pt':float(g.width_pt.iloc[0]),'umap_height_pt':float(g.height_pt.iloc[0]),'complete_multilineage_cohort':TARGET_MCS,'original_five_composition_unchanged':True,'all_20_shown_before_analysis':True,'outputs':panel_records,'source_sha256':sha256(SOURCE)}
(OUT/'reports/validation.json').write_text(json.dumps(qc,indent=2))
display(qc)
display(pseudotime_tests[pseudotime_tests.context.str.startswith('multilineage')])
''')
nb.write(nb.v4.new_notebook(cells=c,metadata={'kernelspec':{'display_name':'chronoclone-paper','language':'python','name':'chronoclone-paper'}}),O/'complete_overview_and_analysis.ipynb')
