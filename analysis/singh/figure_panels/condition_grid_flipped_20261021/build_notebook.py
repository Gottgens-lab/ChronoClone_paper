"""Flipped condition grid: days in columns, metaclone x arm in rows. 2026-10-21.

The authors: "Make a flipped version of those. Do not need color bar / x: day 14 and 27 (two columns) /
y: MC0 [WT, Dmnt3a, Npm1c], MC3 [WT, Dmnt3a, Npm1c], MC7 [WT, Dmnt3a, Npm1c]", then "Make the color
bar in horizontal" and "Just plot the color bar": the key is a separate horizontal colour-bar plot.

Replaces the three separate MC0 / MC3 / MC7 condition grids (arms in columns, days in rows, each with its
own colour bar) with one panel. Setup cells are taken unchanged, except the output directory, kernel name
and execution command, from the audited grid notebook
analysis_workspace/singh/singh_figure2b_grid_layout_20260911/0001_figure2b_grid_layout.ipynb: same registry,
UMAP, bandwidth, common density scale, marker calibration and sparse/empty handling. No new inference.
"""
from pathlib import Path
import nbformat as nb

ROOT = Path('/path/to/ChronoClone')
OUT = ROOT / 'reference_workspace/Figure/singh_condition_grid_flipped_20261021'
SRC = ROOT / 'analysis_workspace/singh/singh_figure2b_grid_layout_20260911/0001_figure2b_grid_layout.ipynb'
STEM = 'figure_6c_condition_grid_flipped'
EXEC_CMD = ("JUPYTER_PATH=/path/to/ChronoClone/.runtime/jupyter "
            "/path/to/ChronoClone/.runtime/chronoclone/bin/jupyter nbconvert "
            "--to notebook --execute --inplace --ExecutePreprocessor.kernel_name=chronoclone-paper "
            f"{OUT}/{STEM}.ipynb")

prior = nb.read(str(SRC), as_version=4)

def adapt(source):
    source = source.replace("OUT=ROOT/'analysis_workspace/singh/singh_figure2b_grid_layout_20260911'",
                            f"OUT=ROOT/'{OUT.relative_to(ROOT)}'")
    source = source.replace("assert Path(sys.executable).parent.parent==ENV",
                            "assert Path(sys.executable).resolve().parent.parent==ENV.resolve()")
    source = source.replace("'kernel':'chronoclone'", "'kernel':'chronoclone-paper'")
    source = source.replace("OUT/'build_grid_notebook.py'", "OUT/'build_notebook.py'")
    for old in [l for l in source.split('\n') if l.startswith('execution_command=')]:
        source = source.replace(old, f"execution_command={EXEC_CMD!r}")
    return source

figure_cell = f'''
# One flipped condition grid: days in columns, metaclone x arm in rows. No colour bar; the key is its own plot below.
stem={STEM!r}
metaclones=[0,3,7];arms=['WT','Dnmt3a','Npm1c'];days=['D14','D27']
rows=[(mc,arm) for mc in metaclones for arm in arms]
nc=len(days);nr=len(rows)
previous_figures=set(plt.get_fignums())
# Explicit geometry in inches: fixed axes size and pitch, so every map is drawn at the same scale.
MC_LABEL=0.34;ARM_LABEL=0.82;AXES=1.15;COL_PITCH=1.22;SUPPORT=0.17;ROW_PITCH=AXES+SUPPORT+0.05;HEADER=0.42
fw=MC_LABEL+ARM_LABEL+COL_PITCH*nc+0.08;fh=HEADER+ROW_PITCH*nr+0.05
fig=plt.figure(figsize=(fw,fh),dpi=160,facecolor='white')
labels=[];support_records=[]
for j,day in enumerate(days):
    labels.append(fig.text((MC_LABEL+ARM_LABEL+COL_PITCH*(j+.5))/fw,1-0.10/fh,'Day '+day[1:],
                           ha='center',va='top',fontsize=14,fontweight='bold',color='#29275f'))
for i,(mc,arm) in enumerate(rows):
    bottom=fh-HEADER-ROW_PITCH*(i+1)+SUPPORT+0.05
    labels.append(fig.text((MC_LABEL+ARM_LABEL-0.08)/fw,(bottom+AXES/2)/fh,arm,ha='right',va='center',
                           fontsize=12,fontweight='bold',color='#29275f'))
    for j,day in enumerate(days):
        selected=cells.loc[cells.mc.eq(mc)&cells.arm.eq(arm)&cells.day.eq(day)].copy()
        n=len(selected);family_count=selected.clone.nunique()
        eligible=n>=MIN_DENSITY_CELLS and family_count>=MIN_DENSITY_FAMILIES
        locked=scale_table.loc[scale_table.mc.eq(mc)&scale_table.arm.eq(arm)&scale_table.day.eq(day)].iloc[0]
        assert (n,family_count)==(locked.cells,locked.families)
        if eligible:
            histogram,_,_=np.histogram2d(selected.umap1,selected.umap2,bins=(X_EDGES,Y_EDGES))
            density=gaussian_filter(histogram/n,sigma=(BANDWIDTH/DX,BANDWIDTH/DY),mode='constant',truncate=4)/(DX*DY)
            assert np.isclose(density.max(),locked.density_peak,rtol=1e-12)
            assert np.isclose(density.sum()*DX*DY,locked.density_integrated_mass,rtol=1e-12)
            point_density=map_coordinates(density,np.vstack(((selected.umap1-X_CENTERS[0])/DX,(selected.umap2-Y_CENTERS[0])/DY)),order=1,mode='nearest')
            order=np.argsort(point_density,kind='stable')
        else:
            point_density=None;order=np.arange(n)
        ax=fig.add_axes([(MC_LABEL+ARM_LABEL+COL_PITCH*j)/fw,bottom/fh,AXES/fw,AXES/fh])
        ax.set_xlim(PLOT_XLIM);ax.set_ylim(PLOT_YLIM);ax.set_aspect('equal');ax.set_axis_off()
        fig.canvas.draw()
        width_pt=ax.get_window_extent().width/fig.dpi*72
        ratio=width_pt/REFERENCE_AXES_WIDTH_PT
        area=15*MARKER_AREA_MULTIPLIER*ratio**2;background_area=2.2*ratio**2;edge=.12*ratio
        ax.scatter(full_umap[:,0],full_umap[:,1],s=background_area,c='#d9dfe4',linewidths=0,alpha=1,rasterized=True,zorder=0)
        if n:
            kwargs={{'c':point_density[order],'cmap':'viridis','norm':DENSITY_NORM}} if eligible else {{'c':SPARSE_COLOR}}
            ax.scatter(selected.umap1.to_numpy()[order],selected.umap2.to_numpy()[order],s=area,
                       **kwargs,edgecolors='#111827',linewidths=edge,alpha=.82,rasterized=False,zorder=2,clip_on=False)
        labels.append(fig.text((MC_LABEL+ARM_LABEL+COL_PITCH*(j+.5))/fw,(bottom-.025)/fh,f'n={{n:,}}  C={{family_count:,}}',
                               ha='center',va='top',fontsize=9,color='#626972'))
        support_records.append({{'stem':stem,'mc':mc,'arm':arm,'day':day,'cells':n,'families':family_count,
                                'representation':locked.representation,'marker_ratio':np.sqrt(area)/width_pt}})
# One metaclone label per group of three arm rows, rotated at the left edge.
for g,mc in enumerate(metaclones):
    top=fh-HEADER-ROW_PITCH*(3*g);low=fh-HEADER-ROW_PITCH*(3*g+3)
    labels.append(fig.text(0.12/fw,(low+top)/2/fh,f'MC{{mc}}',ha='center',va='center',rotation=90,
                           fontsize=15,fontweight='bold',color=MC_LABEL_COLORS[mc]))
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for label in labels:
    box=label.get_window_extent(renderer)
    assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height,(stem,label.get_text(),box)
assert set(plt.get_fignums())-previous_figures=={{fig.number}}
svg=OUT/'panels'/f'{{stem}}.svg';png=svg.with_suffix('.png')
svg.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(svg);fig.savefig(png,dpi=240)
display(fig);plt.close(fig)
support=pd.DataFrame(support_records)
support.to_csv(OUT/'tables'/f'{{stem}}_support.csv',index=False)
display(support)
print(svg,sha256(svg));print('figure size (in):',fw,fh)
'''

KEY_CELL = "\n# The colour key alone, horizontal: same common density scale (DENSITY_NORM) as every map in the grid above.\nstem='figure_6c_density_key_horizontal'\nprevious_figures=set(plt.get_fignums())\nfig=plt.figure(figsize=(2.2,0.62),dpi=160,facecolor='white')\nbar=fig.add_axes([0.10,0.30,0.80,0.16])\ncb=fig.colorbar(plt.cm.ScalarMappable(norm=DENSITY_NORM,cmap='viridis'),cax=bar,orientation='horizontal')\ncb.set_ticks([0,DENSITY_MAX]);cb.set_ticklabels(['Low','High']);cb.ax.tick_params(labelsize=10,length=0,pad=2)\ncb.outline.set_visible(False)\nfig.text(0.5,0.62,'Relative density',fontsize=11,ha='center',va='bottom')\nassert set(plt.get_fignums())-previous_figures=={fig.number}\nsvg=OUT/'panels'/f'{stem}.svg';png=svg.with_suffix('.png')\nfig.savefig(svg);fig.savefig(png,dpi=240)\ndisplay(fig);plt.close(fig)\nprint(svg,sha256(svg),'scale 0 to',DENSITY_MAX)\n"

cells = [nb.v4.new_markdown_cell(f'''# Flipped condition grid — days in columns, metaclone x arm in rows

The authors asked for a flipped version of the MC0 / MC3 / MC7 condition grids: **columns** Day 14 and Day 27,
**rows** MC0 [WT, Dnmt3a, Npm1c], MC3 [WT, Dnmt3a, Npm1c], MC7 [WT, Dnmt3a, Npm1c].
The three separate grids become one panel with no colour bar; the shared density key is drawn as its own
**horizontal** colour-bar plot (The authors: \"Just plot the color bar\").

Display only. The setup cells below are taken unchanged (except the output directory, kernel name and
execution command) from the audited grid notebook
`analysis_workspace/singh/singh_figure2b_grid_layout_20260911/0001_figure2b_grid_layout.ipynb`, which produced
the current condition grids: same cell registry, UMAP, fixed bandwidth, common density scale, marker
calibration and sparse/empty handling. Only the layout changes. No new expression access, embedding,
assignment, test or normalisation.

Environment **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`
(-> `/path/to/ChronoClone/.runtime/chronoclone`), kernel **chronoclone-paper**.
Versions and the execution command are printed by the first cell.
''')]
for idx in [1, 2, 3, 4]:
    cells.append(nb.v4.new_code_cell(adapt(prior.cells[idx].source)))
cells.append(nb.v4.new_code_cell(figure_cell))
cells.append(nb.v4.new_code_cell(KEY_CELL))
notebook = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'chronoclone-paper',
                                                                   'language': 'python',
                                                                   'name': 'chronoclone-paper'}})
nb.write(notebook, str(OUT / f'{STEM}.ipynb'))
print('wrote', OUT / f'{STEM}.ipynb')
