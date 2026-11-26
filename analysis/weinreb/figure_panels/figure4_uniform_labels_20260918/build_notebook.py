"""Redraw Figure 4A-4D with one label size and no target/reference/context labels. 2026-09-18.

The authors: "For figure 4C and 4D, remove the label of 'context', 'target', 'reference'. Make all the labels big
and the same size for all figure 4 panels."
Display only: each panel re-plots the stored clone-level program scores
(analysis_workspace/weinreb/weinreb_metaclone_ancestry_worked_examples_20260910/tables/program_0X_*_plot_data.csv)
and the stored BH q of the registered contrast. Colours, box/whisker convention (Tukey 1.5 IQR, all clones
shown as jittered points), pooled-first order, separator and brackets follow the current panels.
All four panels share one height (2.7 in), one axes height and one text size (9 pt; significance marks 12 pt), drawn at
their final size so the text does not shrink when placed. The pooled column is labelled "Pooled" in every
panel so the 8-column panels fit at this size.
"""
from pathlib import Path
import nbformat as nb

OUT = Path('/path/to/ChronoClone/reference_workspace/Figure/weinreb_figure4_uniform_labels_20260918')
EXEC_CMD = ("JUPYTER_PATH=/path/to/ChronoClone/.runtime/jupyter "
            "/path/to/ChronoClone/.runtime/chronoclone/bin/jupyter nbconvert --to notebook "
            "--execute --inplace --ExecutePreprocessor.kernel_name=chronoclone-paper "
            f"{OUT}/figure4_uniform_labels.ipynb")

setup = f'''
import sys, platform, hashlib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
import matplotlib.pyplot as plt
ROOT = Path('/path/to/ChronoClone')
OUT = ROOT / 'reference_workspace/Figure/weinreb_figure4_uniform_labels_20260918'
SRC = ROOT / 'analysis_workspace/weinreb/weinreb_metaclone_ancestry_worked_examples_20260910/tables'
ENV = Path('/path/to/ChronoClone/.runtime/chronoclone')
assert Path(sys.executable).resolve().parent.parent == ENV.resolve()
(OUT / 'panels').mkdir(parents=True, exist_ok=True); (OUT / 'tables').mkdir(exist_ok=True)
def sha256(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda: s.read(8 * 1024 * 1024), b''): d.update(b)
    return d.hexdigest()
display({{'name': 'chronoclone', 'path': str(ENV), 'manager': 'existing Conda-managed environment; not modified',
         'kernel': 'chronoclone-paper', 'python': platform.python_version(), 'numpy': np.__version__,
         'pandas': pd.__version__, 'matplotlib': mpl.__version__, 'execution_command': {EXEC_CMD!r}}})
FILES = {{'A': 'program_01_mono_1_pooled_and_metaclone_distribution_plot_data.csv',
         'B': 'program_02_mono_2_pooled_and_metaclone_distribution_plot_data.csv',
         'C': 'program_03_multi_divergence_pooled_and_metaclone_distribution_plot_data.csv',
         'D': 'program_04_multi_convergence_pooled_and_metaclone_distribution_plot_data.csv'}}
EFFECTS = SRC / 'primary_program_effects.csv'
PALETTE_SOURCE = ROOT / 'ChronoClone_paper/analysis/weinreb/inputs/manuscript_metaclone_palette.csv'
display(pd.DataFrame([{{'source': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}}
                     for p in [*(SRC / f for f in FILES.values()), EFFECTS, PALETTE_SOURCE]]))
PALETTE = pd.read_csv(PALETTE_SOURCE).set_index('label').color.to_dict()
# Colours of the current panels (read from their SVGs).
ROLE_COLORS = {{'pooled': '#708090', 'context': '#a6adb7', 'reference': '#2166ac', 'target': '#b2182b'}}
# One text size for every label in every Figure 4 panel, drawn at final size.
FONT = 9; MARK_FONT = 12; HEIGHT = 2.7
# Fixed margins in inches, so every panel has the same axes height and label spacing; only the width differs
# (five-column panels 3.6 in, eight-column panels 4.6 in so the 9-pt column labels do not collide).
LEFT, RIGHT, BOTTOM, AXES_HEIGHT = 0.74, 0.10, 0.58, 1.56
mpl.rcParams.update({{'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': FONT,
                     'axes.labelsize': FONT, 'xtick.labelsize': FONT, 'ytick.labelsize': FONT,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'}})
summary_rows = []
'''

TEMPLATE = '''
# Figure 4{letter}: {title}. Full construction in this cell; one output stem.
stem = {stem!r}; ylabel = {ylabel!r}
# Contrast roles as registered for this panel (the stored contrast string names them).
reference = {reference!r}; target = {target!r}
plot_data = pd.read_csv(SRC / FILES[{letter!r}])
assert all(m in plot_data.primary_contrast.iloc[0] for m in reference + target)
assert plot_data.primary_qvalue.nunique() == 1 and plot_data.program.nunique() == 1
q = float(plot_data.primary_qvalue.iloc[0]); mark = '**' if q < .05 else '*' if q < .10 else 'ns'
groups = (plot_data[['display_group', 'display_role', 'display_order', 'group_clone_count']]
          .drop_duplicates().sort_values('display_order').reset_index(drop=True))
assert groups.display_order.tolist() == list(range(len(groups)))
groups['role'] = ['pooled' if r == 'pooled' else 'reference' if g in reference else 'target' if g in target else 'context'
                  for g, r in zip(groups.display_group, groups.display_role)]
rng = np.random.default_rng(20260918)
width = {width}
fig = plt.figure(figsize=(width, HEIGHT), dpi=200)
ax = fig.add_axes([LEFT / width, BOTTOM / HEIGHT, (width - LEFT - RIGHT) / width, AXES_HEIGHT / HEIGHT])
values = [plot_data.loc[plot_data.display_group.eq(g), 'library_adjusted_score'].to_numpy() for g in groups.display_group]
box = ax.boxplot(values, positions=range(len(groups)), widths=.62, whis=1.5, showfliers=False, patch_artist=True,
                 medianprops={{'color': '#202833', 'lw': 1.4}})
for i, role in enumerate(groups.role):
    color = ROLE_COLORS[role]
    box['boxes'][i].set(facecolor=color, alpha=.16, edgecolor='none')
    for part in ('whiskers', 'caps'):
        for line in box[part][2 * i:2 * i + 2]: line.set(color=color, lw=1.1)
    y = values[i]
    ax.scatter(i + rng.uniform(-.17, .17, len(y)), y, s=7, color=color, alpha=.35 if role == 'pooled' else .65,
               lw=0, zorder=3)
ax.axhline(0, color='#b7bec8', lw=.8, zorder=0)
ax.axvline(.5, color='#b7bec8', lw=.8, ls=(0, (3, 3)), zorder=0)
ax.yaxis.grid(True, color='#e8ebef', lw=.6); ax.set_axisbelow(True)
ax.set_xlim(-.6, len(groups) - .4)
ax.set_ylabel(ylabel)
ax.set_xticks(range(len(groups)), [''] * len(groups)); ax.tick_params(axis='x', length=0)
for i, row in groups.iterrows():
    name = 'Pooled' if row.role == 'pooled' else row.display_group
    color = 'black' if row.role == 'pooled' else PALETTE[row.display_group]
    ax.annotate(name, xy=(i, 0), xycoords=('data', 'axes fraction'), xytext=(0, -4), textcoords='offset points',
                ha='center', va='top', fontsize=FONT, color=color)
    ax.annotate(f'n={{int(row.group_clone_count)}}', xy=(i, 0), xycoords=('data', 'axes fraction'), xytext=(0, -15),
                textcoords='offset points', ha='center', va='top', fontsize=FONT)
# Registered contrast: target vs reference metaclones; the pooled column is not tested.
position = dict(zip(groups.display_group, groups.index))
line = {{'color': '#111111', 'lw': 1.0, 'clip_on': False, 'transform': ax.get_xaxis_transform()}}
if len(reference) == 2:
    for members in (reference, target):
        xs = sorted(position[m] for m in members)
        ax.plot([xs[0], xs[0], xs[1], xs[1]], [1.03, 1.08, 1.08, 1.03], **line)
    left, right = sorted([np.mean([position[m] for m in reference]), np.mean([position[m] for m in target])])
    ax.plot([left, left, right, right], [1.10, 1.17, 1.17, 1.10], **line); mark_y = 1.18
else:
    left, right = sorted([position[reference[0]], position[target[0]]])
    ax.plot([left, left, right, right], [1.03, 1.10, 1.10, 1.03], **line); mark_y = 1.11
ax.text((left + right) / 2, mark_y, mark, transform=ax.get_xaxis_transform(), ha='center', va='bottom',
        fontsize=MARK_FONT, fontweight='bold')
svg = OUT / 'panels' / f'{{stem}}.svg'; png = svg.with_suffix('.png')
fig.savefig(svg); fig.savefig(png, dpi=300)
display(fig); plt.close(fig)
summary_rows.append({{'panel': '4{letter}', 'stem': stem, 'program': plot_data.program.iloc[0],
                     'contrast': plot_data.primary_contrast.iloc[0], 'q': q, 'mark': mark,
                     'groups': ', '.join(f'{{g}} (n={{int(n)}})' for g, n in zip(groups.display_group, groups.group_clone_count)),
                     'svg_sha256': sha256(svg)}})
print(svg)
'''

specs = [('A', 'figure_4a_monocyte_inflammatory', 'Inflammatory\nResponse (SD)', 'monocyte metaclones, Inflammatory Response', ['MC6', 'MC8'], ['MC7', 'MC12'], 3.6),
         ('B', 'figure_4b_monocyte_myc_targets_v2', 'MYC Targets V2\n(SD)', 'monocyte metaclones, MYC Targets V2', ['MC6', 'MC8'], ['MC7', 'MC12'], 3.6),
         ('C', 'figure_4c_multilineage_ifn_mc5_mc3', 'Interferon Alpha\nResponse (SD)', 'multilineage metaclones, MC5 vs MC3', ['MC3'], ['MC5'], 4.6),
         ('D', 'figure_4d_multilineage_ifn_mc3_mc18', 'Interferon Alpha\nResponse (SD)', 'multilineage metaclones, MC3 vs MC18', ['MC18'], ['MC3'], 4.6)]

cells = [nb.v4.new_markdown_cell('''# Figure 4A–4D with one label size

The authors: remove the "context", "target" and "reference" labels from 4C and 4D, and make every label big and the
same size across Figure 4. All four panels are redrawn at final size with one height (2.7 in), one axes height and
a single 9-pt text size (significance marks 12 pt). The pooled column is labelled "Pooled" in every panel so
the eight-column panels fit; those two panels are 4.6 in wide (five-column panels 3.6 in) so their 9-pt column
labels do not collide. Y-axis labels wrap onto two lines.

Display only. Each panel re-plots the stored clone-level program scores and the stored BH q of its registered
contrast from `analysis_workspace/weinreb/weinreb_metaclone_ancestry_worked_examples_20260910/tables/`. Colours,
Tukey box/whiskers with every clone shown as a jittered point, pooled-first order, separator and brackets
follow the current panels. No new inference.

Environment **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`
(→ `/path/to/ChronoClone/.runtime/chronoclone`), kernel **chronoclone-paper**.
'''), nb.v4.new_code_cell(setup)]
for letter, stem, ylabel, title, reference, target, width in specs:
    cells.append(nb.v4.new_code_cell(TEMPLATE.format(letter=letter, stem=stem, ylabel=ylabel, title=title,
                                                     reference=reference, target=target, width=width)))
cells.append(nb.v4.new_code_cell('''
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / 'tables/figure4_panel_summary.csv', index=False)
display(summary)
'''))
nb.write(nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'chronoclone-paper',
        'language': 'python', 'name': 'chronoclone-paper'}}), str(OUT / 'figure4_uniform_labels.ipynb'))
print('wrote', OUT / 'figure4_uniform_labels.ipynb')
