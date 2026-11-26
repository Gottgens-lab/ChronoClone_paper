"""Flipped MC0/MC3/MC7 condition grids: days as rows, metaclone x condition as columns.

2026-10-21, at the authors' request ("Make a flipped version of those. Do not need color bar.
x: day 14 and 27 (two rows); y: MC0 [WT, Dnmt3a, Npm1c], MC3 [...], MC7 [...]").

Display only: the setup cells are taken unchanged (except output directory, kernel and execution command)
from the audited grid notebook
analysis_workspace/singh/singh_figure2b_grid_layout_20260911/0001_figure2b_grid_layout.ipynb, which produced the
separate per-metaclone condition grids. Same registry, UMAP, fixed bandwidth, common density scale, marker
calibration and sparse/empty handling; only the layout changes and the colour bar is dropped. No new inference.
"""
from pathlib import Path
import nbformat as nb

ROOT = Path('/path/to/ChronoClone')
OUT = ROOT / 'reference_workspace/Figure/singh_mc_condition_grid_flipped_20261021'
SRC = ROOT / 'analysis_workspace/singh/singh_figure2b_grid_layout_20260911/0001_figure2b_grid_layout.ipynb'
STEM = 'mc0_mc3_mc7_condition_grid_flipped'
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

figure_cell = '''
# One authorized faceted figure: all calculations and full construction visible.
# Flipped layout: Days 14 and 27 are the two rows; the nine columns are MC0, MC3 and MC7, each with
# WT, Dnmt3a and Npm1c. Same per-snapshot density, common colour scale and marker calibration as the
# per-metaclone grids; the colour-bar key is omitted at the authors' request.
stem = 'mc0_mc3_mc7_condition_grid_flipped'
metaclones = [0, 3, 7]; arms = ['WT', 'Dnmt3a', 'Npm1c']; days = ['D14', 'D27']
columns = [(mc, arm) for mc in metaclones for arm in arms]
nc = len(columns); nr = len(days)
previous_figures = set(plt.get_fignums())
fw = 1.05 + 1.55 * nc + 0.25; fh = 1.15 + 1.90 * nr
fig = plt.figure(figsize=(fw, fh), dpi=160, facecolor='white')
labels = []; axes_records = []
for j, (mc, arm) in enumerate(columns):
    centre = (1.05 + 1.55 * (j + .5)) / fw
    if arm == arms[0]:   # one metaclone heading over its three condition columns
        block = (1.05 + 1.55 * (j + len(arms) / 2)) / fw
        labels.append(fig.text(block, 1 - 0.12 / fh, f'MC{mc}', ha='center', va='top',
                               fontsize=16, fontweight='bold', color=MC_LABEL_COLORS[mc]))
    labels.append(fig.text(centre, 1 - 0.55 / fh, arm, ha='center', va='top',
                           fontsize=13, fontweight='bold', color='#29275f'))
for i, day in enumerate(days):
    bottom = fh - 1.15 - 1.9 * (i + 1) + 0.26
    labels.append(fig.text(0.88 / fw, (bottom + .83) / fh, 'Day ' + day[1:], ha='right', va='center',
                           fontsize=13, fontweight='bold', color='#29275f'))
    for j, (mc, arm) in enumerate(columns):
        selected = cells.loc[cells.mc.eq(mc) & cells.arm.eq(arm) & cells.day.eq(day)].copy()
        n = len(selected); family_count = selected.clone.nunique()
        eligible = n >= MIN_DENSITY_CELLS and family_count >= MIN_DENSITY_FAMILIES
        locked = scale_table.loc[scale_table.mc.eq(mc) & scale_table.arm.eq(arm) & scale_table.day.eq(day)].iloc[0]
        assert (n, family_count) == (locked.cells, locked.families)
        if eligible:
            histogram, _, _ = np.histogram2d(selected.umap1, selected.umap2, bins=(X_EDGES, Y_EDGES))
            density = gaussian_filter(histogram / n, sigma=(BANDWIDTH / DX, BANDWIDTH / DY), mode='constant', truncate=4) / (DX * DY)
            assert np.isclose(density.max(), locked.density_peak, rtol=1e-12)
            assert np.isclose(density.sum() * DX * DY, locked.density_integrated_mass, rtol=1e-12)
            point_density = map_coordinates(density, np.vstack(((selected.umap1 - X_CENTERS[0]) / DX,
                                                                (selected.umap2 - Y_CENTERS[0]) / DY)), order=1, mode='nearest')
            order = np.argsort(point_density, kind='stable')
        else:
            point_density = None; order = np.arange(n)
        ax = fig.add_axes([(1.05 + 1.55 * j) / fw, bottom / fh, 1.52 / fw, 1.66 / fh])
        ax.set_xlim(PLOT_XLIM); ax.set_ylim(PLOT_YLIM); ax.set_aspect('equal'); ax.set_axis_off()
        fig.canvas.draw()
        width_pt = ax.get_window_extent().width / fig.dpi * 72
        ratio = width_pt / REFERENCE_AXES_WIDTH_PT
        area = 15 * MARKER_AREA_MULTIPLIER * ratio ** 2; background_area = 2.2 * ratio ** 2; edge = .12 * ratio
        assert np.isclose(np.sqrt(area) / width_pt, np.sqrt(15 * MARKER_AREA_MULTIPLIER) / REFERENCE_AXES_WIDTH_PT, rtol=1e-12)
        ax.scatter(full_umap[:, 0], full_umap[:, 1], s=background_area, c='#d9dfe4',
                   linewidths=0, alpha=1, rasterized=True, zorder=0)
        if n:
            kwargs = {'c': point_density[order], 'cmap': 'viridis', 'norm': DENSITY_NORM} if eligible else {'c': SPARSE_COLOR}
            ax.scatter(selected.umap1.to_numpy()[order], selected.umap2.to_numpy()[order], s=area,
                       **kwargs, edgecolors='#111827', linewidths=edge, alpha=.82, rasterized=False, zorder=2, clip_on=False)
        labels.append(fig.text((1.05 + 1.55 * (j + .5)) / fw, (bottom - .035) / fh, f'n={n:,}  C={family_count:,}',
                               ha='center', va='top', fontsize=11, color='#626972'))
        snapshot_records.append({'stem': stem, 'mc': mc, 'arm': arm, 'day': day, 'cells': n, 'families': family_count,
                                 'representation': locked.representation, 'marker_ratio': np.sqrt(area) / width_pt,
                                 'density_peak': float(locked.density_peak)})
        axes_records.append(ax)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for label in labels:
    box = label.get_window_extent(renderer)
    assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, (stem, label.get_text(), box)
assert set(plt.get_fignums()) - previous_figures == {fig.number}
svg = OUT / 'panels' / f'{stem}.svg'; png = svg.with_suffix('.png')
svg.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(svg); fig.savefig(png, dpi=240)
display(fig); plt.close(fig)
grid_records.append({'stem': stem, 'columns': columns, 'days': days, 'column_dimension': 'mc_x_arm',
                     'width_in': fw, 'height_in': fh, 'svg_sha256': sha256(svg), 'png_sha256': sha256(png)})
print(svg)
'''

cells = [nb.v4.new_markdown_cell('''# MC0, MC3 and MC7 condition grids, flipped into one panel

The authors: "Make a flipped version of those. Do not need color bar. x: day 14 and 27 (two rows);
y: MC0 [WT, Dnmt3a, Npm1c], MC3 [WT, Dnmt3a, Npm1c], MC7 [WT, Dnmt3a, Npm1c]".

So the three separate per-metaclone grids become one panel: **two rows** (Day 14, Day 27) and **nine
columns** — MC0, MC3 and MC7, each with WT, Dnmt3a and Npm1c. The relative-density colour bar is dropped.

Display only. The setup cells below are taken unchanged (except the output directory, kernel name and
execution command) from the audited grid notebook
`analysis_workspace/singh/singh_figure2b_grid_layout_20260911/0001_figure2b_grid_layout.ipynb`: same cell
registry, same UMAP, same fixed bandwidth, same common density scale, same marker calibration, same
sparse/empty handling. No new expression access, embedding, assignment, test or normalisation.

Environment **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`
(→ `/path/to/ChronoClone/.runtime/chronoclone`), kernel **chronoclone-paper**.
Versions and the execution command are printed by the first cell.
''')]
for idx in [1, 2, 3, 4]:
    cells.append(nb.v4.new_code_cell(adapt(prior.cells[idx].source)))
cells.append(nb.v4.new_code_cell(figure_cell))
cells.append(nb.v4.new_code_cell('''
support = pd.DataFrame(snapshot_records)
support.to_csv(OUT / 'tables/condition_grid_flipped_support.csv', index=False)
display(support)
display(pd.DataFrame(grid_records))
'''))
nb.write(nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'chronoclone-paper',
        'language': 'python', 'name': 'chronoclone-paper'}}), str(OUT / f'{STEM}.ipynb'))
print('wrote', OUT / f'{STEM}.ipynb')
