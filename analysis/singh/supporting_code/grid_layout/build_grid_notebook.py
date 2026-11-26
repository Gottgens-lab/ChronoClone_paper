"""Generate explicitly authorized MC-column/day-row display cells; no new inference."""
from pathlib import Path
import nbformat as nb

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
prior=nb.read(ROOT/'analysis_workspace/singh/singh_weinreb_style_umaps_20260911/0001_singh_weinreb_style_umaps.ipynb',as_version=4)
cells=[nb.v4.new_markdown_cell('''# Singh maps in Figure 2B's column-and-row layout

The authors explicitly requested this faceted display: metaclones in columns, days in rows. The complete atlas is split into readable column blocks, separately for WT, Dnmt3a and Npm1c. This authorization is specific to these maps; other manuscript panels remain separate. The shared Day-7 sample appears only under WT. Main comparison examples use the same visual grammar, with conditions in columns when the metaclone is fixed.

This is a display-only revision of the executed spatial atlas. No new expression access, embedding, assignments, tests or density normalization. Coordinates, cell/family support, fixed-bandwidth density, common numerical scale and sparse/empty handling are preserved. Figure 2B supplies the visual reference. The authors subsequently requested larger dots: foreground areas are doubled relative to that calibration, consistently across all grids. Gray context alone is rasterized; foreground points and labels are editable SVG vectors. Only identification, support and scale labels appear inside each figure. Full explanations belong in manuscript legends.

Use existing **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`, Jupyter kernel **chronoclone** (existing Conda-managed installation, not modified). Exact versions and execution command are displayed below. Each output cell contains full snapshot selection, density calculation, faceted construction, one SVG/PNG output stem and display. `build_grid_notebook.py` generates the explicit cells; it does not generate figures itself.
''')]
# Keep audited inputs and visible scientific setup, changing only the output path.
for idx in [1,2,3]:
    source=prior.cells[idx].source.replace('analysis_workspace/singh/singh_weinreb_style_umaps_20260911', 'analysis_workspace/singh/singh_figure2b_grid_layout_20260911')
    source=source.replace("OUT/'build_style_notebook.py'", "OUT/'build_grid_notebook.py'")
    source=source.replace("{OUT}/execute_notebook.py {OUT}/0001_singh_weinreb_style_umaps.ipynb", "{ROOT}/analysis_workspace/singh/singh_weinreb_style_umaps_20260911/execute_notebook.py {OUT}/0001_figure2b_grid_layout.ipynb")
    cells.append(nb.v4.new_code_cell(source))
cells.append(nb.v4.new_code_cell('''
REFERENCE_FIGURE=ROOT/'analysis_workspace/manuscript/chronoclone_sister_clone_revision_20260911/figures/figure_2b_selected_metaclone_maps.svg'
assert REFERENCE_FIGURE.exists()
print('Visual reference:',REFERENCE_FIGURE,sha256(REFERENCE_FIGURE))
REFERENCE_SCRIPT=ROOT/'analysis_workspace/manuscript/chronoclone_manuscript_panel_styling_20260907/build_0001_figure2b_2d_2g_styling_notebook.py'
REFERENCE_BACKGROUND=ROOT/'analysis_workspace/weinreb/weinreb_metaclone_cohesion_power_20260819/tables/iseh2026_registered_cell_state_umap_source.csv'
ref=pd.read_csv(REFERENCE_BACKGROUND,usecols=['cell_id','UMAP1','UMAP2']).drop_duplicates('cell_id')
reference_aspect=(ref.UMAP1.max()-ref.UMAP1.min()+1.6)/(ref.UMAP2.max()-ref.UMAP2.min()+1.6)
# Actual Figure 2B GridSpec geometry, followed by equal-aspect adjustment.
reference_box_width=7.2*(.995-.10)/(5+4*.02)
reference_box_height=7.8*(.94-.025)/(2+.33)/(3-2*.04)
REFERENCE_AXES_WIDTH_PT=min(reference_box_width,reference_box_height*reference_aspect)*72
print('Current Figure 2B axes width (pt):',REFERENCE_AXES_WIDTH_PT)
MARKER_AREA_MULTIPLIER=2.0  # the authors' subsequent explicit request: make dots bigger.
print('Foreground diameter multiplier over Figure 2B:',np.sqrt(MARKER_AREA_MULTIPLIER))
display(pd.DataFrame([{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for p in [REFERENCE_FIGURE,REFERENCE_SCRIPT,REFERENCE_BACKGROUND]]))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.hashsalt':'singh_figure2b_grid_20260911'})
grid_records=[];snapshot_records=[]
print('Authorized layout: metaclone columns, day rows; separate condition blocks; Day 7 only under WT.')
'''))

specs=[('figure_5d_wt_histories_grid',[0,3,7],['WT'],['D7','D14','D27'],'mc'),
       ('figure_6c_mc03_condition_grid',[3],['WT','Dnmt3a','Npm1c'],['D14','D27'],'arm'),
       ('figure_6g_mc00_condition_grid',[0],['WT','Dnmt3a','Npm1c'],['D14','D27'],'arm')]
blocks=[[0,1,2,3,4],[5,6,7,9],[10,11,12,13],[15,16,17,19]]
for arm in ['WT','Dnmt3a','Npm1c']:
    for i,mcs in enumerate(blocks,1):
        specs.append((f'atlas_{arm.lower()}_block_{i:02d}',mcs,[arm],['D7','D14','D27'] if arm=='WT' else ['D14','D27'],'mc'))
template=r'''
# One authorized faceted figure: all calculations and full construction visible.
stem=STEM;metaclones=MCS;arms=ARMS;days=DAYS;column_dimension=DIM
columns=metaclones if column_dimension=='mc' else arms
nc=len(columns);nr=len(days)
previous_figures=set(plt.get_fignums())
fw=1.05+1.55*nc+0.95;fh=0.8+1.90*nr
fig=plt.figure(figsize=(fw,fh),dpi=160,facecolor='white')
labels=[];axes_records=[]
group_label=arms[0] if column_dimension=='mc' else f'MC{metaclones[0]}'
labels.append(fig.text((1.05+1.55*nc/2)/fw,1-0.12/fh,group_label,
                      ha='center',va='top',fontsize=16,fontweight='bold',
                      color='#29275f' if column_dimension=='mc' else MC_LABEL_COLORS[metaclones[0]]))
for j,column in enumerate(columns):
    label=f'MC{column}' if column_dimension=='mc' else column
    labels.append(fig.text((1.05+1.55*(j+.5))/fw,1-0.43/fh,label,
                 ha='center',va='top',fontsize=14,fontweight='bold',
                 color=MC_LABEL_COLORS[column] if column_dimension=='mc' else '#29275f'))
for i,day in enumerate(days):
    bottom=fh-0.8-1.9*(i+1)+0.26
    labels.append(fig.text(0.88/fw,(bottom+.83)/fh,'Day '+day[1:],ha='right',va='center',
                          fontsize=13,fontweight='bold',color='#29275f'))
    for j,column in enumerate(columns):
        mc=column if column_dimension=='mc' else metaclones[0]
        arm=arms[0] if column_dimension=='mc' else column
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
        ax=fig.add_axes([(1.05+1.55*j)/fw,bottom/fh,1.52/fw,1.66/fh])
        ax.set_xlim(PLOT_XLIM);ax.set_ylim(PLOT_YLIM);ax.set_aspect('equal');ax.set_axis_off()
        fig.canvas.draw()
        width_pt=ax.get_window_extent().width/fig.dpi*72
        ratio=width_pt/REFERENCE_AXES_WIDTH_PT
        area=15*MARKER_AREA_MULTIPLIER*ratio**2;background_area=2.2*ratio**2;edge=.12*ratio
        assert np.isclose(np.sqrt(area)/width_pt,np.sqrt(15*MARKER_AREA_MULTIPLIER)/REFERENCE_AXES_WIDTH_PT,rtol=1e-12)
        ax.scatter(full_umap[:,0],full_umap[:,1],s=background_area,c='#d9dfe4',
                   linewidths=0,alpha=1,rasterized=True,zorder=0)
        if n:
            kwargs={'c':point_density[order],'cmap':'viridis','norm':DENSITY_NORM} if eligible else {'c':SPARSE_COLOR}
            ax.scatter(selected.umap1.to_numpy()[order],selected.umap2.to_numpy()[order],s=area,
                       **kwargs,edgecolors='#111827',linewidths=edge,alpha=.82,rasterized=False,zorder=2,clip_on=False)
        labels.append(fig.text((1.05+1.55*(j+.5))/fw,(bottom-.035)/fh,f'n={n:,}  C={family_count:,}',
                               ha='center',va='top',fontsize=11,color='#626972'))
        snapshot_records.append({'stem':stem,'mc':mc,'arm':arm,'day':day,'cells':n,'families':family_count,
                                 'representation':locked.representation,'marker_ratio':np.sqrt(area)/width_pt,
                                 'density_peak':float(locked.density_peak)})
        axes_records.append(ax)
# Integral common-scale key, not a separate analytical panel.
bar=fig.add_axes([(1.05+1.55*nc+.15)/fw,.32,.11/fw,.33])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=DENSITY_NORM,cmap='viridis'),cax=bar)
cb.set_ticks([0,DENSITY_MAX]);cb.set_ticklabels(['Low','High']);cb.ax.tick_params(labelsize=10,length=0,pad=3)
cb.outline.set_visible(False)
labels.append(fig.text((1.05+1.55*nc+.15)/fw,.675,'Relative\ndensity',fontsize=11,ha='left',va='bottom'))
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for label in labels:
    box=label.get_window_extent(renderer)
    assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height,(stem,label.get_text(),box)
assert set(plt.get_fignums())-previous_figures=={fig.number}
svg=OUT/'panels'/f'{stem}.svg';png=svg.with_suffix('.png')
fig.savefig(svg);fig.savefig(png,dpi=240)
display(fig);plt.close(fig)
grid_records.append({'stem':stem,'columns':columns,'days':days,'column_dimension':column_dimension,
                     'width_in':fw,'height_in':fh,'svg_sha256':sha256(svg),'png_sha256':sha256(png)})
print(svg)
'''
for stem,mcs,arms,days,dim in specs:
    code=template.replace('STEM',repr(stem)).replace('MCS',repr(mcs)).replace('ARMS',repr(arms)).replace('DAYS',repr(days)).replace('DIM',repr(dim))
    cells.append(nb.v4.new_code_cell(code))
cells.append(nb.v4.new_code_cell('''
grid_table=pd.DataFrame(grid_records);snapshots=pd.DataFrame(snapshot_records)
complete=snapshots.loc[snapshots.stem.str.startswith('atlas_')].copy()
assert len(grid_table)==15 and len(complete)==119
assert not complete[['mc','arm','day']].duplicated().any()
comparison=complete.sort_values(['mc','arm','day']).reset_index(drop=True)
expected=scale_table.sort_values(['mc','arm','day']).reset_index(drop=True)
pd.testing.assert_frame_equal(comparison[['mc','arm','day','cells','families','representation']],expected[['mc','arm','day','cells','families','representation']])
assert not complete.loc[complete.day.eq('D7'),'arm'].ne('WT').any()
assert np.allclose(snapshots.marker_ratio,np.sqrt(15*MARKER_AREA_MULTIPLIER)/REFERENCE_AXES_WIDTH_PT,rtol=1e-12)
grid_table.to_json(OUT/'tables/grid_manifest.json',orient='records',indent=2)
snapshots.to_csv(OUT/'tables/grid_snapshot_manifest.csv',index=False)
qc={'complete_atlas_snapshots':len(complete),'grid_figures':len(grid_table),'same_cohort_counts':True,
    'same_coordinates':True,'same_density_estimates_and_common_scale':True,'consistent_marker_proportions_across_grids':True,
    'foreground_area_multiplier_over_figure2b':MARKER_AREA_MULTIPLIER,
    'day7_only_wt':True,'no_new_inference':True,'visual_review':'pending individual review'}
(OUT/'reports/grid_integrity.json').write_text(json.dumps(qc,indent=2))
display(grid_table);display(qc)
'''))
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'chronoclone','language':'python','name':'chronoclone'}})
nb.write(book,HERE/'0001_figure2b_grid_layout.ipynb')
print(HERE/'0001_figure2b_grid_layout.ipynb')
