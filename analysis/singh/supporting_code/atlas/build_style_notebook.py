"""Build explicit, source-preserving Singh UMAP cells with the author Weinreb style."""
from pathlib import Path
import nbformat as nb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'analysis_workspace/singh/singh_landscape_expansion_20260911/atlas/0001_metaclone_spatial_atlas.ipynb'
book = nb.read(SOURCE, as_version=4)
book.cells[0] = nb.v4.new_markdown_cell('''# Singh per-metaclone, per-day UMAPs in the author Weinreb style

## Scope

This is a visual-style revision of the corrected Singh spatial atlas, not a new biological analysis. Each map is a separate raw panel. The 17 metaclones have seven genuine snapshots each: shared WT Day 7, then WT, Npm1c and Dnmt3a at Days 14 and 27. Shared Day 7 is not duplicated as an observed mutant sample. Every cell, family assignment, stored UMAP coordinate, sampling count, density calculation and common density normalization is retained from the existing atlas.

## Visual reference and size calibration

The author reference is `reference_workspace/notebooks/Figure2_klein_pace.ipynb`, zero-based cell 44, producing `reference_workspace/Figure/fig2/panels/panel_2i_density_all_mc_flipped.svg`. The original marker settings are foreground area 50 pt², alpha 0.8, black edge width 0.15 pt; background area 3.5 pt², lightgrey, no edge. Original equal-aspect map axes are 104.590775 pt wide, obtained from the 31-inch, 20-column GridSpec and corroborated by its saved SVG geometry. Each new map measures its own **post-equal-aspect axes width**. Marker areas scale by the squared width ratio and outline widths by the width ratio. Thus circle diameter relative to the displayed UMAP matches the original; copying `s=50` into a larger standalone figure would not match it.

The gray background alone is rasterized in SVG. Foreground circles, outlines and labels remain editable vectors. Foreground markers are not clipped at the invisible axes boundary: full-circle glyphs may extend into the white plot margin, and their clearance from the figure frame is checked separately. Maps have equal aspect, identical coordinate limits, low-to-high density drawing order, white background and no axis furniture. Only identification and sample-size labels appear on each panel. The source metaclone label colors are unchanged.

## Color and sampling interpretation

Only the appearance changes. Singh retains its fixed-bandwidth (0.4 UMAP units), 256 × 256 Gaussian-smoothed within-snapshot probability density and the same global color scale across all maps. This is not the per-metaclone-normalized local-neighborhood count used by the Weinreb reference. Do not compare numerical density values between datasets. Color denotes spatial concentration, not expansion, cell recovery, or independent biological replication. Density requires at least 20 cells and three resolved families; other nonempty snapshots retain fixed navy points. Gray-only maps with zero observed cells do not establish extinction. `n` denotes observed cells and `C` denotes resolved families. HSPCs remain pooled.

## Environment and reproducibility

Use existing environment **chronoclone**, absolute path `/path/to/ChronoClone/.runtime/chronoclone`, Jupyter kernel **chronoclone**. This existing Conda-managed environment is reused without modification. Exact versions and the absolute execution command are displayed below. `build_style_notebook.py` generates the explicit cells in this notebook from the prior executed atlas; every plot cell visibly includes its own filtering, density calculation, construction, SVG/PNG export and display. Neither author files nor the previous atlas are modified.
''')
old_out = 'analysis_workspace/singh/singh_landscape_expansion_20260911/atlas'
new_out = 'analysis_workspace/singh/singh_weinreb_style_umaps_20260911'
for cell in book.cells:
    if cell.cell_type != 'code':
        continue
    cell.outputs = []
    cell.execution_count = None
    cell.source = cell.source.replace(old_out, new_out)
    cell.source = cell.source.replace(
        "{ROOT}/analysis_workspace/manuscript/chronoclone_panel_revision_20260910/execute_notebook.py {OUT}/0001_metaclone_spatial_atlas.ipynb",
        "{OUT}/execute_notebook.py {OUT}/0001_singh_weinreb_style_umaps.ipynb")
    cell.source = cell.source.replace("'font.family':'Arial'", "'font.family':'DejaVu Sans'")
    cell.source = cell.source.replace("'font_family':'Arial'", "'font_family':'DejaVu Sans'")
    cell.source = cell.source.replace("'svg.hashsalt':'singh_atlas_20260911'", "'svg.hashsalt':'singh_weinreb_style_20260911'")

# Display choices only. Scientific setup is retained verbatim from the executed atlas.
book.cells[1].source += '''
AUTHOR_NOTEBOOK=ROOT/'reference_workspace/notebooks/Figure2_klein_pace.ipynb'
AUTHOR_SVG=ROOT/'reference_workspace/Figure/fig2/panels/panel_2i_density_all_mc_flipped.svg'
OLD_ATLAS=ROOT/'analysis_workspace/singh/singh_landscape_expansion_20260911/atlas'
REFERENCE_AXES_WIDTH_PT=31 * 72 * (0.995-0.04) / (20+19*0.02)
assert abs(REFERENCE_AXES_WIDTH_PT-104.590775)<1e-6
style_sources=pd.DataFrame([{'source':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)}
    for p in [AUTHOR_NOTEBOOK,AUTHOR_SVG,OLD_ATLAS/'0001_metaclone_spatial_atlas.ipynb',OUT/'build_style_notebook.py']])
style_sources.to_csv(OUT/'tables/style_source_manifest.csv',index=False)
display(style_sources)
print('Reference axes width (pt):',REFERENCE_AXES_WIDTH_PT)
'''
book.cells[3].source += '''
# Numeric display quantities and all cohort rows must remain unchanged from the previous atlas.
old_scale=pd.read_csv(OLD_ATLAS/'tables/snapshot_support_and_density_scale.csv')
pd.testing.assert_frame_equal(scale_table,old_scale,check_exact=False,rtol=1e-12,atol=1e-14)
old_coordinates=pd.read_csv(OLD_ATLAS/'tables/atlas_cell_coordinates.csv.gz')
pd.testing.assert_frame_equal(cells[['cell_id','mc','arm','day','clone','umap1','umap2']],old_coordinates,
                              check_exact=False,rtol=1e-14,atol=1e-14)
style={'reference_axes_width_pt':REFERENCE_AXES_WIDTH_PT,'foreground_reference_area_pt2':50,
       'background_reference_area_pt2':3.5,'foreground_reference_edge_pt':0.15,
       'foreground_alpha':0.8,'background_color':'#d3d3d3','font_family':'DejaVu Sans',
       'size_rule':'area = reference area * (actual equal-aspect axes width / reference width)^2',
       'density_calculation_and_common_scale_unchanged':True,'coordinate_limits_unchanged':True}
design.update(style)
(OUT/'reports/design.json').write_text(json.dumps(design,indent=2))
display(style)
'''

map_count = 0
for cell in book.cells:
    if cell.cell_type != 'code' or '# One individual raw atlas map:' not in cell.source:
        continue
    map_count += 1
    old = "ax.scatter(full_umap[:,0],full_umap[:,1],s=1.7,c='#D6D6D6',linewidths=0,alpha=1,rasterized=True)"
    assert old in cell.source
    cell.source = cell.source.replace(old, '''# Measure the actual equal-aspect plotting area before calibrating marker dimensions.
ax.set_xlim(PLOT_XLIM);ax.set_ylim(PLOT_YLIM);ax.set_aspect('equal');ax.set_axis_off()
fig.canvas.draw()
axes_width_pt=ax.get_window_extent().width / fig.dpi * 72
linear_scale=axes_width_pt/REFERENCE_AXES_WIDTH_PT
foreground_area=50*linear_scale**2
background_area=3.5*linear_scale**2
edge_width=0.15*linear_scale
assert np.isclose(np.sqrt(foreground_area)/axes_width_pt,np.sqrt(50)/REFERENCE_AXES_WIDTH_PT,rtol=1e-12)
assert np.isclose(np.sqrt(background_area)/axes_width_pt,np.sqrt(3.5)/REFERENCE_AXES_WIDTH_PT,rtol=1e-12)
ax.scatter(full_umap[:,0],full_umap[:,1],s=background_area,c='lightgrey',linewidths=0,
           alpha=1,rasterized=True,zorder=0)''')
    cell.source = cell.source.replace('s=24,', 's=foreground_area,')
    cell.source = cell.source.replace('linewidths=0,alpha=1,rasterized=False)',
        "edgecolors='black',linewidths=edge_width,alpha=0.8,rasterized=False,zorder=2,clip_on=False)")
    cell.source = cell.source.replace("'single_new_figure':True,'label_bounds_passed':True", """'single_new_figure':True,'label_bounds_passed':True,
                       'axes_width_pt':axes_width_pt,'foreground_area_pt2':foreground_area,
                       'background_area_pt2':background_area,'edge_width_pt':edge_width,
                       'axes_left_pt':ax.get_window_extent().x0/fig.dpi*72,
                       'axes_bottom_pt':ax.get_window_extent().y0/fig.dpi*72,
                       'axes_height_pt':ax.get_window_extent().height/fig.dpi*72,
                       'figure_width_pt':fig.get_figwidth()*72,'figure_height_pt':fig.get_figheight()*72,
                       'foreground_clip_on':False,
                       'marker_diameter_to_axes_width':np.sqrt(foreground_area)/axes_width_pt,
                       'marker_ratio_matches_author':True""")
assert map_count == 119
book.cells[-1].source += '''
assert render_table.marker_ratio_matches_author.all()
assert np.allclose(render_table.marker_diameter_to_axes_width,np.sqrt(50)/REFERENCE_AXES_WIDTH_PT,rtol=1e-12)
assert all(sha256(ROOT/r.source)==r.sha256 for r in style_sources.itertuples())
old_manifest=pd.read_csv(OLD_ATLAS/'tables/atlas_panel_manifest.csv')
pd.testing.assert_frame_equal(render_table[['stem','mc','arm','day','cells','families','representation']],
                              old_manifest[['stem','mc','arm','day','cells','families','representation']])
qc.update({'all_marker_diameter_ratios_match_author':True,'scientific_display_values_unchanged':True,
           'all_119_counts_and_representations_unchanged':True,'author_and_previous_atlas_unchanged':True,
           'style_source_manifest':'tables/style_source_manifest.csv'})
(OUT/'reports/atlas_integrity.json').write_text(json.dumps(qc,indent=2))
display(qc)
'''
nb.validate(book)
HERE.mkdir(parents=True, exist_ok=True)
target = HERE / '0001_singh_weinreb_style_umaps.ipynb'
nb.write(book, target)
print(target)
