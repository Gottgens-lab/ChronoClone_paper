"""Build the independent, executed evidence review of the ancestry companion."""
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

# Filled with exact image hashes after direct inspection of the completed run.
# These are human/agent review provenance, not computed scientific conclusions.
REVIEWED_PROGRAM_PNGS = {
    'program_01_mono_1_pooled_and_metaclone_distribution.png': 'cb561103f011fb46bfdfe07470af6da6801ce616c2537473210ce071cfb74fc4',
    'program_02_mono_2_pooled_and_metaclone_distribution.png': '4eda55c1cc81991cbe46b902f9cd6ec6a397ba1643593af2be0a24ac0f7bf928',
    'program_03_multi_divergence_pooled_and_metaclone_distribution.png': 'a259ce6e8bbad23d47fb4bdf7c0fe75190485f016666e931e2625b4b56b90080',
    'program_04_multi_convergence_pooled_and_metaclone_distribution.png': '57adb1307e42b02dedc8e9cb4e8e3da202e23f3860c1caa5ed769c75e895357d',
    'program_05_monocyte_all50_row_zscore_heatmap.png': '4df2932e05210b809cd353753626b0ac2bdf1749d7bffc4aad95f85ab8f82467',
    'program_06_mono_1_sensitivity_forest.png': 'b3898d620e39059a2b2c42b1a1cec2145f1d2ba7cbf5aabcad8e6fd3639bef35',
    'program_07_multi_divergence_sensitivity_forest.png': 'd1312d134d16967b4cf92e5c0a8fdef542cc208e17eb25b72f4d96ee3060bf3c',
    'program_08_mono_1_leading_gene_contributions.png': 'fda625655ef17e329810f217abb4386dc47256ac922d00a20918b5184265918f',
    'program_09_multi_divergence_leading_gene_contributions.png': '50b3ab7da8f3ddf4d26a484bae0170d4a30bc6dea602bde9e860b246456a4347',
}
REVIEWED_DYNAMICS_PNGS = {
    'ancestry_01_day2_clone_linkage.png': '5310645a9f55f86ee688257985ff7a0afd96877121e4490c5f42fbf273694aa4',
    'ancestry_02_monocyte_branch_position.png': '6f0271b812ab259e3bd0eca20ec4bafcdab06324f6823fcaeb8d275922af627c',
    'ancestry_03_multilineage_branch_occupancy_change.png': '114766d5749c3db1d6fbde26ab4b22a2375d4d42662c5badac08a592f8665598',
}
DYNAMICS_REVIEWER = "research /root/ancestry_dynamics agent; individual post-rerun approval communicated by /root on 2026-09-10"


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [md(r"""
# Weinreb ancestry worked examples: independent evidence review

## Goal and scope

Review the completed `0001_weinreb_metaclone_ancestry_worked_examples.ipynb`
without refitting its scientific models. This companion independently checks
multiple testing with SciPy, reconstructs the pace contrast from the original
adjusted means, reconciles retained contrasts, verifies cell/clone display grain,
checks saved sensitivity intervals and cNMF boundaries, and audits executed
notebook and editable-figure provenance. This is a computational and presentation
review of the same data, not independent biological validation.

## Environment and execution

Use the existing **chronoclone** environment at
`/path/to/ChronoClone/.runtime/chronoclone`, Jupyter kernel
`chronoclone`. No environment is installed or mutated. Exact local executables,
Python/package versions and the execution command are printed below.

Every finding below is calculated in executed cells. Manual figure reviews are
identified separately, attributed to their actual reviewer and tied to exact PNG
hashes. The review creates no new biological tests, gene programs or figures.
"""), code(r"""
import ast
import hashlib
import importlib.metadata as im
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from scipy import stats
from IPython.display import display

PROJECT_ROOT = Path('/path/to/ChronoClone')
ANALYSIS_DIR = PROJECT_ROOT / 'analysis_workspace/analyses/weinreb_metaclone_ancestry_worked_examples_20260910'
SOURCE_DIR = PROJECT_ROOT / 'analysis_workspace/analyses/weinreb_day2_hspc_metaclone_gene_programs_20260902'
FIGDIR, TABLEDIR, REPORTDIR = [ANALYSIS_DIR / x for x in ['figures', 'tables', 'reports']]
REVIEW_NOTEBOOK = ANALYSIS_DIR / '0002_evidence_review.ipynb'
MAIN_NOTEBOOK = ANALYSIS_DIR / '0001_weinreb_metaclone_ancestry_worked_examples.ipynb'
ENV_PATH = Path('/path/to/ChronoClone/.runtime/chronoclone')
EXECUTION_COMMAND = f'env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {ENV_PATH}/bin/jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=chronoclone --ExecutePreprocessor.timeout=7200 {REVIEW_NOTEBOOK}'
assert Path(sys.executable).resolve() == (ENV_PATH / 'bin/python').resolve()
assert (ENV_PATH / 'bin/jupyter-nbconvert').exists()
environment = pd.DataFrame({'field': ['environment_name', 'environment_manager', 'environment_path', 'kernel', 'python', 'execution_command'] + ['version_' + p for p in ['numpy', 'pandas', 'scipy', 'nbformat', 'nbconvert', 'ipykernel']],
                            'value': ['chronoclone', 'Existing Conda-compatible environment; reused without mutation', str(ENV_PATH), 'chronoclone', sys.version.split()[0], EXECUTION_COMMAND] + [im.version(p) for p in ['numpy', 'pandas', 'scipy', 'nbformat', 'nbconvert', 'ipykernel']]})
environment.to_csv(REPORTDIR / 'evidence_review_environment.csv', index=False)
display(environment)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

checks = []
def check(name, condition, detail):
    checks.append({'check': name, 'passed': bool(condition), 'detail': str(detail)})

primary = pd.read_csv(TABLEDIR / 'primary_program_effects.csv')
sensitivity = pd.read_csv(TABLEDIR / 'program_sensitivity_effects.csv.gz')
selection = pd.read_csv(TABLEDIR / 'selected_program_examples.csv')
metadata = pd.read_csv(TABLEDIR / 'clone_metadata.csv', index_col=0)
metadata.index = metadata.index.astype(str)
adjusted_scores = pd.read_csv(TABLEDIR / 'clone_library_adjusted_program_scores.csv.gz', index_col=0)
adjusted_scores.index = adjusted_scores.index.astype(str)
original = pd.read_csv(SOURCE_DIR / 'tables/hallmark_program_contrasts.csv')
original_means = pd.read_csv(SOURCE_DIR / 'tables/adjusted_program_means_by_metaclone.csv').set_index('metaclone')
source_manifest = pd.read_csv(ANALYSIS_DIR / 'provenance/source_manifest.csv')
manifest_hash_check = pd.DataFrame([{'path': row.path, 'sha256_expected': row.sha256, 'sha256_observed': sha256(PROJECT_ROOT / row.path),
                                    'hash_agrees': sha256(PROJECT_ROOT / row.path) == row.sha256,
                                    'size_agrees': (PROJECT_ROOT / row.path).stat().st_size == row.bytes}
                                   for row in source_manifest.itertuples(index=False)])
manifest_hash_check.to_csv(REPORTDIR / 'evidence_review_source_hash_audit.csv', index=False)
check('frozen_source_hashes_and_sizes', manifest_hash_check.hash_agrees.all() and manifest_hash_check.size_agrees.all(), len(manifest_hash_check))
check('one_primary_observation_per_LARRY_clone', metadata.index.is_unique and metadata.clone.astype(str).eq(metadata.index).all(), len(metadata))
check('primary_hypothesis_keys_unique', not primary.duplicated(['program', 'contrast_id']).any(), len(primary))
display(manifest_hash_check[['path', 'hash_agrees', 'size_agrees']])
"""), md(r"""
## Independently reconstruct multiple-testing families and the pace contrast

SciPy's `false_discovery_control(method="bh")` supplies an independent
implementation of BH. The original within-lineage OVR and pairwise families
remain intact, including unsupported estimates. The newly declared pace contrast
has its own fixed Hallmark family. Sensitivity families retain missing estimands
as p=1; unsupported estimates are not silently interpreted as null findings.
"""), code(r"""
expected_family_sizes = {'within_lineage_ovr': 950, 'within_lineage_pairwise': 2000, 'monocyte_pace_50': 50}
bh_rows = []
for tier, frame in primary.groupby('tier', sort=True):
    independent_q = stats.false_discovery_control(frame.pvalue.to_numpy(float), method='bh')
    error = float(np.max(np.abs(independent_q - frame.qvalue.to_numpy(float))))
    bh_rows.append({'scope': 'primary', 'variant': 'primary', 'tier': tier, 'n_tests': len(frame),
                    'expected_n_tests': expected_family_sizes[tier], 'max_q_error': error,
                    'q_lt_005': int((independent_q < .05).sum()),
                    'q_005_to_010': int(((independent_q >= .05) & (independent_q < .10)).sum())})
    check('independent_primary_BH_' + tier, len(frame) == expected_family_sizes[tier] and error < 1e-12, error)
for (variant, tier), frame in sensitivity.groupby(['variant', 'tier'], sort=True):
    independent_q = stats.false_discovery_control(frame.pvalue.fillna(1).to_numpy(float), method='bh')
    error = float(np.max(np.abs(independent_q - frame.fit_tier_fdr.to_numpy(float))))
    bh_rows.append({'scope': 'sensitivity', 'variant': variant, 'tier': tier, 'n_tests': len(frame),
                    'expected_n_tests': expected_family_sizes[tier], 'max_q_error': error,
                    'q_lt_005': int((independent_q < .05).sum()),
                    'q_005_to_010': int(((independent_q >= .05) & (independent_q < .10)).sum())})
bh_audit = pd.DataFrame(bh_rows)
check('independent_sensitivity_BH_and_fixed_family_sizes',
      bh_audit.n_tests.eq(bh_audit.expected_n_tests).all() and bh_audit.max_q_error.lt(1e-12).all(),
      f'{len(bh_audit)} primary/sensitivity families')
bh_audit.to_csv(TABLEDIR / 'evidence_review_independent_BH.csv', index=False)
display(bh_audit[bh_audit.scope == 'primary'])

kept = original[original.tier.isin(['within_lineage_ovr', 'within_lineage_pairwise'])].copy()
joined = primary[primary.tier.isin(['within_lineage_ovr', 'within_lineage_pairwise'])].merge(
    kept[['program', 'contrast_id', 'effect_sd', 'standard_error', 'pvalue', 'tier_fdr']],
    on=['program', 'contrast_id'], how='outer', suffixes=('', '_original'), indicator=True, validate='one_to_one')
reconciliation = pd.DataFrame([{'quantity': field, 'max_absolute_error': float(np.max(np.abs(joined[field] - joined[old_field])))}
                               for field, old_field in [('effect_sd', 'effect_sd_original'), ('standard_error', 'standard_error_original'),
                                                       ('pvalue', 'pvalue_original'), ('qvalue', 'tier_fdr')]])
check('all_original_2950_hypotheses_reconciled', len(joined) == 2950 and joined['_merge'].eq('both').all(), len(joined))
check('original_effect_SE_p_q_reproduced', reconciliation.max_absolute_error.lt(1e-8).all(), reconciliation.to_dict('records'))
reconciliation.to_csv(TABLEDIR / 'evidence_review_original_contrast_reconciliation.csv', index=False)
pace = primary[primary.contrast_id == 'Monocyte_fast_minus_slow'].copy().set_index('program')
independent_pace = (original_means.loc[[7, 12], pace.index].mean(axis=0)
                    - original_means.loc[[6, 8], pace.index].mean(axis=0))
pace_audit = pd.DataFrame({'program': pace.index, 'effect_from_original_adjusted_means': independent_pace.loc[pace.index].to_numpy(),
                           'fitted_pace_effect': pace.effect_sd.to_numpy(), 'qvalue': pace.qvalue.to_numpy()})
pace_audit['absolute_error'] = np.abs(pace_audit.effect_from_original_adjusted_means - pace_audit.fitted_pace_effect)
check('pace_effect_reconstructed_from_original_MC_means', len(pace_audit) == 50 and pace_audit.absolute_error.lt(1e-8).all(), pace_audit.absolute_error.max())
pace_audit.to_csv(TABLEDIR / 'evidence_review_pace_contrast_reconstruction.csv', index=False)
cross_pair_effects = pd.DataFrame({f'MC{fast}_minus_MC{slow}': original_means.loc[fast, pace.index] - original_means.loc[slow, pace.index]
                                  for fast in [7, 12] for slow in [6, 8]})
independent_rank = pd.DataFrame({'program': pace.index, 'abs_effect': np.abs(independent_pace.loc[pace.index]).to_numpy(),
                                 'all_cross_pair_same_direction': (cross_pair_effects.gt(0).all(axis=1) | cross_pair_effects.lt(0).all(axis=1)).to_numpy()})
independent_rank = independent_rank.sort_values(['all_cross_pair_same_direction', 'abs_effect', 'program'], ascending=[False, False, True])
rank_selection = [{'example_id': f'mono_{rank}', 'independently_selected_program': row.program,
                    'selection_source': 'four original adjusted-mean cross-pair directions, then absolute equal-MC effect; no p/q input'}
                   for rank, row in enumerate(independent_rank.head(2).itertuples(index=False), 1)]
for example_id, target, reference in [('multi_divergence', 5, 3), ('multi_convergence', 3, 18)]:
    original_pair_effects = original_means.loc[target, pace.index] - original_means.loc[reference, pace.index]
    independent_pair_rank = pd.DataFrame({'program': pace.index, 'abs_effect': np.abs(original_pair_effects).to_numpy()}).sort_values(['abs_effect', 'program'], ascending=[False, True])
    rank_selection.append({'example_id': example_id, 'independently_selected_program': independent_pair_rank.program.iloc[0],
                           'selection_source': 'largest absolute original adjusted-mean pair effect; no p/q input'})
selection_rank_audit = pd.DataFrame(rank_selection).merge(selection[['example_id', 'program']], on='example_id', validate='one_to_one')
selection_rank_audit['selection_agrees'] = selection_rank_audit.independently_selected_program.eq(selection_rank_audit.program)
check('example_selection_reconstructed_without_p_or_q_input', len(selection_rank_audit) == 4 and len(independent_rank) == 50
      and selection_rank_audit.selection_agrees.all(), selection_rank_audit[['example_id', 'program', 'selection_agrees']].to_dict('records'))
selection_rank_audit.to_csv(TABLEDIR / 'evidence_review_example_selection.csv', index=False)
display(reconciliation)
display(pace_audit.sort_values('qvalue').head(10))
display(selection_rank_audit)
"""), md(r"""
## Check selected examples, sensitivity intervals and same-clone displays

This section verifies the selected rows against the complete effect tables,
including source sensitivity support. Pooled-family distribution columns must
contain exactly the same clones and score values as their MC-resolved columns.
The gene charts must sum to the fitted program contrast and are not interpreted
as gene-level tests.
"""), code(r"""
selected_rows = []
for example in selection.itertuples(index=False):
    row = primary[(primary.program == example.program) & (primary.contrast_id == example.contrast_id)].iloc[0]
    variant_rows = sensitivity[(sensitivity.program == example.program) & (sensitivity.contrast_id == example.contrast_id)]
    source = variant_rows[variant_rows.variant == 'source_fraction_adjusted'].iloc[0]
    prog2 = variant_rows[variant_rows.variant == 'prog2_only'].iloc[0]
    nonmito = variant_rows[variant_rows.variant == 'nonmito_poscounts'].iloc[0]
    loo = variant_rows[variant_rows.variant.str.startswith('without_library__')]
    supported_loo = loo[loo.support_gate & loo.estimable & loo.effect_primary_scale.notna()]
    same_direction = int((np.sign(supported_loo.effect_primary_scale) == np.sign(row.effect_sd)).sum())
    selected_rows.append({'example_id': example.example_id, 'program': example.program, 'contrast_id': example.contrast_id,
                          'effect_sd': row.effect_sd, 'qvalue': row.qvalue, 'ci_low': row.ci_low, 'ci_high': row.ci_high,
                          'effect_and_q_match': bool(np.isclose(example.effect_sd, row.effect_sd, atol=1e-12) and np.isclose(example.qvalue, row.qvalue, atol=1e-12)),
                          'source_fraction_effect': source.effect_sd, 'source_fraction_support': bool(source.support_gate),
                          'source_summary_matches': bool(np.isclose(example.source_fraction_effect_sd, source.effect_sd, atol=1e-12)),
                          'prog2_only_effect_fixed_scale': prog2.effect_primary_scale, 'prog2_supported': bool(prog2.support_gate),
                          'prog2_summary_matches': bool(example.prog2_supported == prog2.support_gate and np.isclose(example.prog2_effect_primary_scale, prog2.effect_primary_scale, atol=1e-12, equal_nan=True)),
                          'nonmito_effect_fixed_scale': nonmito.effect_primary_scale, 'nonmito_supported': bool(nonmito.support_gate),
                          'nonmito_native_qvalue': nonmito.fit_tier_fdr,
                          'nonmito_summary_matches': bool(example.nonmito_supported == nonmito.support_gate and np.isclose(example.nonmito_effect_primary_scale, nonmito.effect_primary_scale, atol=1e-12, equal_nan=True)
                                                        and np.isclose(example.nonmito_native_qvalue, nonmito.fit_tier_fdr, atol=1e-12, equal_nan=True)),
                          'LOO_supported': len(supported_loo), 'LOO_same_direction': same_direction,
                          'LOO_summary_matches': bool(example.loo_supported == len(supported_loo) and example.loo_same_direction == same_direction)})
selected_audit = pd.DataFrame(selected_rows)
check('all_four_selected_examples_reconciled', len(selected_audit) == 4 and selected_audit.filter(like='match').all().all(), len(selected_audit))
check('all_ten_sensitivity_variants_present', sensitivity.variant.nunique() == 10 and 'nonmito_poscounts' in set(sensitivity.variant), sorted(sensitivity.variant.unique()))
selected_audit.to_csv(TABLEDIR / 'evidence_review_selected_examples.csv', index=False)
display(selected_audit)

normalization_audit = pd.read_csv(REPORTDIR / 'normalization_support_audit.csv')
nonmito_normalization = normalization_audit[normalization_audit.variant == 'nonmito_poscounts'].iloc[0]
nonmito_genes = str(nonmito_normalization.genes).split('|')
size_factors = pd.read_csv(TABLEDIR / 'normalization_size_factors_by_clone.csv')
check('nonmitochondrial_normalization_gene_register', len(nonmito_genes) == int(nonmito_normalization.normalization_genes)
      and not any(g.lower().startswith('mt-') for g in nonmito_genes), len(nonmito_genes))
check('normalization_size_factor_support_per_clone', len(size_factors) == len(metadata) and size_factors.clone.astype(str).is_unique
      and set(size_factors.clone.astype(str)) == set(metadata.index)
      and np.isfinite(size_factors.nonmito_poscounts_size_factor).all() and size_factors.nonmito_poscounts_size_factor.gt(0).all()
      and size_factors.nonmito_normalization_genes_used.gt(0).all()
      and np.isclose(np.exp(np.mean(np.log(size_factors.nonmito_poscounts_size_factor))), 1, atol=1e-12), len(size_factors))
normalization_compatibility = pd.read_csv(REPORTDIR / 'normalization_implementation_equivalence.csv')
check('executed_main_PyDESeq2_normalization_compatibility_audit',
      normalization_compatibility.size_factor_direct_vs_api_max_error.lt(1e-12).all()
      and normalization_compatibility.score_genes_unchanged.eq(9777).all()
      and normalization_compatibility.clones_unchanged.eq(len(metadata)).all()
      and normalization_compatibility.eligible_normalization_genes.eq(len(nonmito_genes)).all(), normalization_compatibility.to_dict('records'))
display(normalization_compatibility)

distribution_audit = []
distribution_paths = sorted(TABLEDIR.glob('program_0[1-4]_*_distribution_plot_data.csv'))
for path in distribution_paths:
    frame = pd.read_csv(path, dtype={'clone_id': str})
    example = selection[selection.example_id == frame.example_id.iloc[0]].iloc[0]
    pooled = frame[frame.display_role == 'pooled']
    resolved = frame[frame.display_role == 'metaclone']
    expected_clones = set(metadata.index[metadata.lineage_bias == example.context])
    twin = pooled.set_index('clone_id').library_adjusted_score.sort_index().sub(resolved.set_index('clone_id').library_adjusted_score.sort_index())
    source_error = np.max(np.abs(resolved.set_index('clone_id').library_adjusted_score - adjusted_scores.loc[resolved.clone_id, example.program].set_axis(resolved.clone_id)))
    ok = (set(pooled.clone_id) == set(resolved.clone_id) == expected_clones
          and pooled.clone_id.is_unique and resolved.clone_id.is_unique
          and frame.groupby('clone_id').size().eq(2).all()
          and twin.abs().lt(1e-12).all() and source_error < 1e-12
          and resolved.display_group.eq(resolved.metaclone.map(lambda mc: f'MC{int(mc)}')).all())
    distribution_audit.append({'plot_table': path.name, 'unique_clones': len(expected_clones), 'display_rows': len(frame),
                               'duplicate_score_max_error': float(twin.abs().max()), 'source_score_max_error': float(source_error),
                               'same_clone_display_verified': bool(ok)})
distribution_audit = pd.DataFrame(distribution_audit)
check('all_four_distributions_repeat_exact_same_clones', len(distribution_audit) == 4 and distribution_audit.same_clone_display_verified.all(), len(distribution_audit))
distribution_audit.to_csv(TABLEDIR / 'evidence_review_distribution_grain.csv', index=False)
display(distribution_audit)

forest_checks = []
for path in sorted(TABLEDIR.glob('program_0[6-7]_*_sensitivity_forest_plot_data.csv')):
    frame = pd.read_csv(path)
    delta = stats.norm.ppf(.975) * frame.primary_scale_se
    interval_error = max(float(np.nanmax(np.abs(frame.ci_low_primary_scale - (frame.effect_primary_scale - delta)))),
                         float(np.nanmax(np.abs(frame.ci_high_primary_scale - (frame.effect_primary_scale + delta)))))
    supported = frame.support_gate & frame.estimable & frame.effect_primary_scale.notna() & frame.primary_scale_se.notna()
    forest_checks.append({'plot_table': path.name, 'max_interval_error': interval_error,
                          'support_display_matches': frame.display_supported.eq(supported).all(),
                          'unsupported_rows_have_reason': frame.loc[~supported, 'undefined_reason'].notna().all()})
forest_checks = pd.DataFrame(forest_checks)
check('frozen_scale_forest_CIs_and_NE_are_exact', len(forest_checks) == 2 and forest_checks.max_interval_error.lt(1e-12).all()
      and forest_checks.support_display_matches.all() and forest_checks.unsupported_rows_have_reason.all(), forest_checks.to_dict('records'))
forest_checks.to_csv(TABLEDIR / 'evidence_review_sensitivity_forests.csv', index=False)

contributions = pd.read_csv(TABLEDIR / 'program_leading_gene_contributions.csv')
gene_sums = contributions.groupby('example_id').program_sd_contribution.sum().rename('sum_gene_contributions').to_frame().join(selection.set_index('example_id').effect_sd)
gene_sums['absolute_error'] = np.abs(gene_sums.sum_gene_contributions - gene_sums.effect_sd)
check('all_gene_contributions_sum_to_program_effect', len(gene_sums) == 4 and gene_sums.absolute_error.lt(1e-9).all(), gene_sums.absolute_error.max())
for path in sorted(TABLEDIR.glob('program_0[8-9]_*_leading_gene_contributions_plot_data.csv')):
    frame = pd.read_csv(path)
    example_id = frame.example_id.iloc[0]
    check('displayed_gene_contributions_' + example_id, np.isclose(frame.program_sd_contribution.sum(), gene_sums.loc[example_id, 'effect_sd'], atol=1e-9)
          and int(frame.n_genes_represented.sum()) == int((contributions.example_id == example_id).sum()), path.name)
gene_sums.to_csv(TABLEDIR / 'evidence_review_gene_contribution_sums.csv')
display(gene_sums)
"""), md(r"""
## Execution, inline plot construction and editable SVG checks

Inspect the executed main notebook independently of its own assertions. Each plot
must shape its own `plot_data`, create one figure, save one SVG stem and retain
one visible image output. Its recorded runtime figure/save counts must agree with
the parsed source and actual editable SVG. Plot symbols and colors do not create
new statistical tests.
"""), code(r"""
main_notebook = nbformat.read(MAIN_NOTEBOOK, as_version=4)
nbformat.validate(main_notebook)
code_cells = [(i, cell) for i, cell in enumerate(main_notebook.cells) if cell.cell_type == 'code']
execution_counts = [cell.execution_count for _, cell in code_cells]
errors = [(i, output.get('ename', 'error')) for i, cell in code_cells for output in cell.outputs if output.output_type == 'error']
stderr = [(i, output.get('text', '')) for i, cell in code_cells for output in cell.outputs if output.output_type == 'stream' and output.get('name') == 'stderr']
stderr_summary = pd.DataFrame([
    {'notebook_cell_index': i, 'stderr_streams': sum(index == i for index, _ in stderr),
     'stderr_lines': sum(len(text.splitlines()) for index, text in stderr if index == i),
     'unique_messages': ' | '.join(sorted({line for index, text in stderr if index == i for line in text.splitlines()}))}
    for i in sorted({index for index, _ in stderr})
], columns=['notebook_cell_index', 'stderr_streams', 'stderr_lines', 'unique_messages'])
stderr_summary.to_csv(REPORTDIR / 'evidence_review_main_stderr_audit.csv', index=False)
check('main_notebook_sequential_complete_execution', execution_counts == list(range(1, len(code_cells) + 1)), execution_counts)
check('main_notebook_no_error_outputs', not errors, errors)
check('main_notebook_no_stderr_outputs', not stderr, stderr_summary.to_dict('records'))
main_notebook_provenance = pd.DataFrame([{'path': str(MAIN_NOTEBOOK.relative_to(PROJECT_ROOT)),
                                        'bytes': MAIN_NOTEBOOK.stat().st_size, 'sha256': sha256(MAIN_NOTEBOOK),
                                        'code_cells': len(code_cells), 'error_outputs': len(errors), 'stderr_streams': len(stderr)}])
main_notebook_provenance.to_csv(REPORTDIR / 'evidence_review_main_notebook_provenance.csv', index=False)
runtime = pd.read_csv(REPORTDIR / 'plot_runtime_audit.csv')
plot_source_rows = []
for index, cell in code_cells:
    tree = ast.parse(cell.source)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    stem_nodes = [node for node in assignments if any(isinstance(target, ast.Name) and target.id == 'plot_stem' for target in node.targets)]
    if not stem_nodes:
        continue
    stem = ast.literal_eval(stem_nodes[0].value)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    creation = [node for node in calls if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'plt' and node.func.attr == 'subplots']
    svg_saves = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'savefig'
                 and node.args and any(isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.endswith('.svg') for child in ast.walk(node.args[0]))]
    shaping = [node for node in assignments if any(isinstance(target, ast.Name) and target.id == 'plot_data' for target in node.targets)]
    no_hidden_plot_function = not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
    images = [output for output in cell.outputs if output.output_type in ['display_data', 'execute_result']
              and any(mime in output.get('data', {}) for mime in ['image/png', 'image/svg+xml', 'image/jpeg'])]
    recorded = runtime[runtime.svg_stems.astype(str).map(lambda value: Path(value).stem == stem)]
    inline_ok = len(creation) == 1 and len(svg_saves) == 1 and shaping and min(node.lineno for node in shaping) < creation[0].lineno and no_hidden_plot_function
    runtime_ok = (len(recorded) == 1 and int(recorded.iloc[0].figures_created) == 1
                  and int(recorded.iloc[0].svg_saves) == 1 and int(recorded.iloc[0].notebook_cell_index) == index)
    svg = FIGDIR / (stem + '.svg')
    png = FIGDIR / (stem + '.png')
    editable = svg.exists() and '<image' not in svg.read_text()
    plot_source_rows.append({'notebook_cell_index': index, 'execution_count': cell.execution_count, 'plot_stem': stem,
                             'inline_plot_data_before_figure': bool(inline_ok), 'source_figure_calls': len(creation),
                             'source_svg_save_calls': len(svg_saves), 'visible_image_outputs': len(images),
                             'runtime_one_figure_one_SVG': bool(runtime_ok), 'editable_SVG_without_raster': bool(editable),
                             'png_exists': png.exists(), 'svg_sha256': sha256(svg) if svg.exists() else '',
                             'png_sha256': sha256(png) if png.exists() else ''})
plot_audit = pd.DataFrame(plot_source_rows)
check('twelve_plots_inline_runtime_and_visible_outputs', len(plot_audit) == len(runtime) == 12
      and plot_audit.inline_plot_data_before_figure.all() and plot_audit.runtime_one_figure_one_SVG.all()
      and plot_audit.visible_image_outputs.eq(1).all(), len(plot_audit))
check('all_SVGs_editable_and_matched_to_plot_cells', plot_audit.editable_SVG_without_raster.all()
      and set(path.stem for path in FIGDIR.glob('*.svg')) == set(plot_audit.plot_stem), len(plot_audit))
plot_audit.to_csv(REPORTDIR / 'evidence_review_inline_plot_and_execution_audit.csv', index=False)
display(plot_audit[['execution_count', 'plot_stem', 'inline_plot_data_before_figure', 'visible_image_outputs', 'runtime_one_figure_one_SVG', 'editable_SVG_without_raster']])
main_validation = pd.read_csv(REPORTDIR / 'validation.csv')
dynamics_validation = pd.read_csv(TABLEDIR / 'ancestry_dynamics_validation.csv')
check('main_own_validation_passes', main_validation.passed.all(), len(main_validation))
check('dynamics_own_validation_passes', dynamics_validation.passed.all(), len(dynamics_validation))

heat = pd.read_csv(TABLEDIR / 'program_05_monocyte_all50_row_zscore_heatmap_plot_data.csv')
heat_wide = heat.pivot(index='program', columns='metaclone', values='adjusted_program_mean_sd').reindex(columns=[6, 8, 7, 12])
expected_z = heat_wide.sub(heat_wide.mean(axis=1), axis=0).div(heat_wide.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
heat_indexed = heat.set_index(['program', 'metaclone'])
expected_long = expected_z.stack(future_stack=True).reindex(heat_indexed.index)
check('heatmap_population_zscores_and_clipping', len(heat) == 200 and np.allclose(expected_long, heat_indexed.raw_zscore, atol=1e-12, equal_nan=True)
      and np.allclose(np.clip(heat.raw_zscore, -2.5, 2.5), heat.display_zscore, atol=1e-12, equal_nan=True)
      and heat.zscore_ddof.eq(0).all(), len(heat))
heat_svg = (FIGDIR / 'program_05_monocyte_all50_row_zscore_heatmap.svg').read_text().lower()
check('heatmap_blue_white_red_contract_retained', all(color in heat_svg for color in ['#2166ac', '#ffffff', '#b2182b']), 'SVG palette endpoints and zero')
"""), md(r"""
## Preserve the independent cNMF sensitivity and its boundaries

The original selected programs are not refitted or reselected here. Recompute
the saved recovery and conservative novelty Boolean decisions from their recorded
diagnostics and thresholds, and verify that the original cNMF findings remain
verbatim in the companion's computed report.
"""), code(r"""
recovery = pd.read_csv(SOURCE_DIR / 'tables/cnmf_postfreeze_program_recovery_summary.csv')
expected_recovery = (
    recovery.n_refits.eq(recovery.expected_refits)
    & recovery.n_finite_matched_spectrum_cosine.eq(recovery.expected_refits)
    & recovery.n_finite_usage_spearman_rho.eq(recovery.expected_refits)
    & recovery.minimum_matched_spectrum_cosine.ge(.75)
    & recovery.median_usage_spearman_rho.ge(.50)
    & recovery.minimum_primary_common_spectral_norm_fraction.ge(np.sqrt(.5))
    & recovery.minimum_refit_common_spectral_norm_fraction.ge(np.sqrt(.5)))
check('cNMF_recovery_gates_reproduced_without_reset', expected_recovery.eq(recovery.recovery_supported).all(), len(recovery))
novelty = pd.read_csv(SOURCE_DIR / 'tables/cnmf_metaclone_program_novelty_assessment.csv')
expected_novel = (novelty.metaclone_association_supported & ~novelty.captured_by_clone_score_correlation
                  & ~novelty.captured_by_gene_overlap & novelty.library_recovery_supported
                  & novelty.source_balance_recovery_supported & novelty.complete_supported_stability
                  & novelty.source_covariate_direction_agrees & ~novelty.any_technical_or_context_flag)
check('cNMF_conservative_novelty_gate_reproduced', expected_novel.eq(novelty.novel_not_captured_candidate).all(), len(novelty))
check('cNMF_zero_conservative_candidates_preserved', int(expected_novel.sum()) == 0, int(expected_novel.sum()))
original_cnmf_text = (SOURCE_DIR / 'reports/cnmf_findings_and_boundaries.txt').read_text()
current_findings = (REPORTDIR / 'computed_findings.txt').read_text()
check('cNMF_computed_findings_verbatim', original_cnmf_text in current_findings, sha256(SOURCE_DIR / 'reports/cnmf_findings_and_boundaries.txt'))
cnmf_summary = recovery.groupby('sensitivity').agg(programs=('program', 'size'), recovery_supported=('recovery_supported', 'sum')).reset_index()
cnmf_summary['conservative_novel_candidates'] = int(expected_novel.sum())
cnmf_summary.to_csv(TABLEDIR / 'evidence_review_preserved_cNMF_summary.csv', index=False)
display(cnmf_summary)

main_manifest = pd.read_csv(REPORTDIR / 'artifact_manifest.csv')
manifest_checks = pd.DataFrame([{'path': r.path, 'hash_matches': sha256(PROJECT_ROOT / r.path) == r.sha256,
                                 'size_matches': (PROJECT_ROOT / r.path).stat().st_size == r.bytes}
                                for r in main_manifest.itertuples(index=False)])
check('completed_main_artifact_manifest_hashes', manifest_checks.hash_matches.all() and manifest_checks.size_matches.all(), len(manifest_checks))
manifest_checks.to_csv(REPORTDIR / 'evidence_review_main_artifact_manifest_audit.csv', index=False)
""")]

manual_source = r"""
# Direct manual reviews are immutable hash-tied provenance, separate from tests.
reviewed_program_pngs = __PROGRAM_REVIEWS__
reviewed_dynamics_pngs = __DYNAMICS_REVIEWS__
dynamics_reviewer = __DYNAMICS_REVIEWER__
manual_rows = []
for filename, expected_hash in reviewed_program_pngs.items():
    path = FIGDIR / filename
    notes = 'Individually inspected: labels, counts, source grain, axes and statistical annotation readable; no overlaps or clipping.'
    if filename.startswith('program_02_'):
        notes += ' MYC Targets V2 visibly carries the proliferation-associated context flag.'
    if filename.startswith('program_05_'):
        notes += ' All 50 rows readable; negative-blue/zero-white/positive-red scale and distinct q symbols retained.'
    if filename.startswith('program_07_'):
        notes += ' Unsupported prog_2-only comparison explicitly displays NE.'
    manual_rows.append({'filename': filename, 'reviewer': 'research program_value_review agent: direct individual image inspection',
                        'review_scope': 'direct program-figure visual QA', 'reviewed_png_sha256': expected_hash,
                        'current_png_sha256': sha256(path), 'review_matches_current_file': sha256(path) == expected_hash, 'notes': notes})
for filename, expected_hash in reviewed_dynamics_pngs.items():
    path = FIGDIR / filename
    manual_rows.append({'filename': filename, 'reviewer': dynamics_reviewer,
                        'review_scope': 'delegated dynamics-figure visual QA; approval attributed to named reviewer',
                        'reviewed_png_sha256': expected_hash, 'current_png_sha256': sha256(path),
                        'review_matches_current_file': sha256(path) == expected_hash,
                        'notes': 'Approved individually by the attributed dynamics reviewer; root supplied the review and exact image hashes.'})
manual_qa = pd.DataFrame(manual_rows)
check('manual_reviews_cover_exact_current_program_figures', len(reviewed_program_pngs) == 9
      and manual_qa[manual_qa.review_scope == 'direct program-figure visual QA'].review_matches_current_file.all(), len(reviewed_program_pngs))
check('attributed_dynamics_reviews_cover_exact_current_figures', len(reviewed_dynamics_pngs) == 3
      and manual_qa[manual_qa.review_scope.str.startswith('delegated')].review_matches_current_file.all(), dynamics_reviewer)
check('manual_reviews_cover_each_executed_plot_once', not manual_qa.filename.duplicated().any()
      and set(manual_qa.filename) == set(plot_audit.plot_stem + '.png'), len(manual_qa))
manual_qa.to_csv(REPORTDIR / 'evidence_review_manual_visual_QA.csv', index=False)
display(manual_qa[['filename', 'reviewer', 'review_matches_current_file', 'notes']])

validation = pd.DataFrame(checks)
validation.to_csv(REPORTDIR / 'evidence_review_validation.csv', index=False)
summary = [
    'INDEPENDENT EVIDENCE REVIEW: WEINREB ANCESTRY WORKED EXAMPLES',
    f"Checks passed: {int(validation.passed.sum())}/{len(validation)}.",
    f'Main notebook: {len(code_cells)} sequentially executed code cells; {len(errors)} error outputs; {len(stderr)} stderr streams; {len(plot_audit)} individually constructed and visible figures.',
    f'Original retained contrasts: {len(joined)}; largest effect/SE/p/q reconciliation error {reconciliation.max_absolute_error.max():.3g}.',
    f'Independent SciPy BH reconciliation: {len(bh_audit)} primary/sensitivity families; maximum q error {bh_audit.max_q_error.max():.3g}.',
    f'Pace contrast reconstructed independently from original adjusted MC means: maximum error {pace_audit.absolute_error.max():.3g}.',
    f'Same-clone pooled/resolved display audits: {len(distribution_audit)}; each clone appears exactly twice with unchanged score.',
    f'Manual image reviews tied to current hashes: {int(manual_qa.review_matches_current_file.sum())}/{len(manual_qa)}.',
    f'Preserved conservative uncaptured cNMF candidates: {int(expected_novel.sum())}.',
    'Observed selected examples:'
]
for r in selected_audit.itertuples(index=False):
    summary.append(f'  {r.example_id}: {r.program}, effect {r.effect_sd:+.4f} SD, q={r.qvalue:.6g}; source-fraction effect {r.source_fraction_effect:+.4f}; prog2-only supported={r.prog2_supported}; nonmitochondrial poscounts effect {r.nonmito_effect_fixed_scale:+.4f} in primary units, native q={r.nonmito_native_qvalue:.6g}; supported same-direction library removals {r.LOO_same_direction}/{r.LOO_supported}.')
normalization_audit = pd.read_csv(REPORTDIR / 'normalization_support_audit.csv')
summary += [
    f"Normalization support: {int(normalization_audit.loc[normalization_audit.variant.eq('primary'), 'normalization_genes'].iloc[0])} universally positive genes in the primary ratio normalization; nonmitochondrial positive-count sensitivity uses {len(nonmito_genes)} eligible genes with {int(size_factors.nonmito_normalization_genes_used.min())}–{int(size_factors.nonmito_normalization_genes_used.max())} contributing genes per clone.",
    'Interpretation: reproducible retrospective clone-level associations conditional on the in-sample partition and retained normalization. Library removal is a sensitivity, not independent biological replication.',
    'Multilineage source adjustment and unsupported source restriction constrain claims of metaclone-specific mechanisms. MYC Targets V2 is a proliferation-associated curated set.',
    'No new scientific model was fitted by this review; no source data, manuscript, or prior analysis was modified.'
]
(REPORTDIR / 'evidence_review_findings.txt').write_text('\n'.join(summary) + '\n')
print('\n'.join(summary))
display(validation)
assert validation.passed.all(), validation.loc[~validation.passed].to_dict('records')
review_artifacts = sorted(list(TABLEDIR.glob('evidence_review_*')) + list(REPORTDIR.glob('evidence_review_*')))
review_manifest = pd.DataFrame([{'path': str(p.relative_to(PROJECT_ROOT)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                                for p in review_artifacts if p.name != 'evidence_review_artifact_manifest.csv'])
review_manifest.to_csv(REPORTDIR / 'evidence_review_artifact_manifest.csv', index=False)
"""
cells += [md("""
## Attributed visual QA and computed review findings

The manual review register records only images actually inspected, or an explicitly
attributed delegated approval. A hash mismatch invalidates that approval for the
current image. The following output combines the independent calculation checks
with the scientific limitations already visible in the companion.
"""), code(manual_source.replace('__PROGRAM_REVIEWS__', repr(REVIEWED_PROGRAM_PNGS))
                        .replace('__DYNAMICS_REVIEWS__', repr(REVIEWED_DYNAMICS_PNGS))
                        .replace('__DYNAMICS_REVIEWER__', repr(DYNAMICS_REVIEWER)))]

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'chronoclone', 'language': 'python', 'name': 'chronoclone'},
    'language_info': {'name': 'python', 'version': '3.11.15'}})
nbf.validate(notebook)
path = ROOT / '0002_evidence_review.ipynb'
nbf.write(notebook, path)
print(path)
