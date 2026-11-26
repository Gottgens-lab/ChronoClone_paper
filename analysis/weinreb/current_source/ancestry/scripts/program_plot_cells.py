"""Notebook-native molecular figures for the frozen ancestry worked examples.

All plot-specific transformations and figure construction are written into each
returned code cell. Source-generation loops never run as notebook plotting loops.
"""
import textwrap

import nbformat as nbf


def build_program_plot_cells():
    cells = [nbf.v4.new_markdown_cell(textwrap.dedent("""
    ## Molecular worked examples: pooled and metaclone-resolved views

    These figures use the fixed program ranking above. A pooled family and its
    constituent metaclones show the same clones twice; the two views are not
    independent groups and are not compared by a hypothesis test. Distributions
    show clone-level scores after subtracting the fitted library contribution,
    in the original 569-clone score units. Boxes show the median and interquartile
    range, whiskers extend to 1.5 interquartile ranges, and dots retain every clone.

    The complete monocyte heatmap uses all fixed Hallmarks and a population
    z-score across the four adjusted MC means. Red means a positive row z-score,
    white zero, and blue a negative score. Symbols refer to the original
    within-family OVR multiple-testing family, not the heatmap colors.

    Sensitivity forests compare the same contrast on frozen primary gene/program
    scales with pointwise 95% intervals. Unsupported comparisons are labelled NE.
    Leading-gene charts algebraically decompose the selected program effect; they
    do not introduce gene-level significance tests.
    """).strip())]

    distribution_source = r'''
    # Chart contract: clone-score distributions; pooled and resolved views repeat
    # the same cells' clone aggregates. No test compares the repeated x groups.
    example_id = __EXAMPLE__
    plot_stem = __STEM__
    example = selection.loc[selection.example_id == example_id].iloc[0]
    program = str(example.program)
    family = str(example.context)
    context_flag = 'proliferation-associated curated program' if program.upper().replace('_', ' ').endswith('MYC TARGETS V2') else ''
    mc_order = list(MONO_ORDER if family == 'Monocyte' else MULTI_ORDER)
    frame = metadata.loc[metadata.metaclone.isin(mc_order)].copy()
    frame['library_adjusted_score'] = adjusted_scores.loc[frame.index, program]
    frame['clone_id'] = frame.index.astype(str)
    frame['program'] = program
    frame['example_id'] = example_id
    frame['prog2_fraction'] = frame.n_prog_2_cells / frame.n_cells
    pooled = frame.assign(display_group=family + ' pooled', display_role='pooled', display_order=0)
    resolved = frame.assign(display_group=frame.metaclone.map(lambda x: f'MC{int(x)}'),
                            display_role='metaclone', display_order=frame.metaclone.map({mc: i + 1 for i, mc in enumerate(mc_order)}))
    plot_data = pd.concat([pooled, resolved], ignore_index=True)
    plot_data['repeated_observation_display'] = True
    plot_data['primary_contrast'] = str(example.contrast_label)
    plot_data['primary_qvalue'] = float(example.qvalue)
    plot_data['primary_q_family'] = '50-Hallmark pace BH' if family == 'Monocyte' else 'original 2,000-test pairwise BH'
    plot_data['program_context_flag'] = context_flag
    assert plot_data.groupby('clone_id').size().eq(2).all()
    assert len(frame) == (135 if family == 'Monocyte' else 217)
    groups = [family + ' pooled'] + [f'MC{mc}' for mc in mc_order]
    target_mcs = [int(x[2:]) for x in str(example.target_levels).split('|')]
    reference_mcs = [int(x[2:]) for x in str(example.reference_levels).split('|')]
    colors = ['#708090'] + ['#B2182B' if mc in target_mcs else '#2166AC' if mc in reference_mcs else '#A6ADB7' for mc in mc_order]
    labels = [f'{family}\npooled\nn={len(frame)}']
    for mc in mc_order:
        role = ('slower' if mc in [6, 8] else 'faster') if family == 'Monocyte' else ('target' if mc in target_mcs else 'reference' if mc in reference_mcs else 'context')
        labels.append(f'MC{mc}\n{role}\nn={int((frame.metaclone == mc).sum())}')
    plot_data['group_clone_count'] = plot_data.groupby('display_group').clone_id.transform('nunique')
    plot_data.to_csv(TABLEDIR / f'{plot_stem}_plot_data.csv', index=False)

    fig, ax = plt.subplots(figsize=(10.0 if family == 'Monocyte' else 12.0, 6.3))
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.255, top=0.795)
    rng = np.random.default_rng(20260910 + __SEED__)
    for i, group in enumerate(groups):
        values = plot_data.loc[plot_data.display_group == group, 'library_adjusted_score'].to_numpy(float)
        boxes = ax.boxplot([values], positions=[i], widths=0.51, whis=1.5,
                          showfliers=False, patch_artist=True, manage_ticks=False,
                          medianprops={'color': '#202833', 'linewidth': 1.5},
                          whiskerprops={'color': colors[i], 'linewidth': 1.1},
                          capprops={'color': colors[i], 'linewidth': 1.1},
                          boxprops={'facecolor': colors[i], 'edgecolor': colors[i], 'alpha': 0.17}, zorder=2)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(values)), values,
                   color=colors[i], s=17 if i else 13, alpha=0.60 if i else 0.29,
                   edgecolors='none', zorder=3)
    ax.axvline(0.5, color='#B7BEC8', linewidth=1.0, linestyle=(0, (3, 3)))
    ax.axhline(0, color='#AEB6C2', linewidth=0.8, zorder=1)
    ax.set_xticks(np.arange(len(groups)), labels, fontsize=10)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel('Library-adjusted program score\n(original clone-level SD units)', fontsize=11)
    ax.grid(axis='y', color='#E8EBEF', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0, pad=8)
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    title = program.replace('HALLMARK_', '').replace('_', ' ')
    fig.text(0.10, 0.955, title, ha='left', va='top', fontsize=17, weight='bold', color='#202833')
    fig.text(0.10, 0.902, f'Day-2 HSPCs · {family} family · {len(frame)} LARRY clones · library-adjusted clone scores',
             ha='left', va='top', fontsize=11, color='#4D5868')
    if context_flag:
        fig.text(0.10, 0.848, 'Context flag: proliferation-associated curated program; no unique metaclone mechanism established.',
                 ha='left', va='top', fontsize=10.1, color='#6B4C55')
    q = float(example.qvalue)
    q_tier = 'q < 0.05' if q < 0.05 else 'exploratory 0.05 ≤ q < 0.10' if q < 0.10 else 'q ≥ 0.10'
    q_family = '50-Hallmark pace BH' if family == 'Monocyte' else 'original 2,000-test pairwise BH'
    fig.text(0.10, 0.122, f'Labelled contrast: {example.contrast_label}; effect {example.effect_sd:+.2f} SD; {q_family} q={q:.3g} ({q_tier}).',
             ha='left', va='top', fontsize=9.2, color='#283444')
    fig.text(0.10, 0.081, 'The pooled column repeats every clone shown in the MC columns; these views are not independent groups.',
             ha='left', va='top', fontsize=9.2, color='#4D5868')
    fig.text(0.10, 0.045, 'Dots: clones. Boxes: median/IQR; whiskers: 1.5 IQR. Illustration ranked by effect size, not q value.',
             ha='left', va='top', fontsize=9.2, color='#4D5868')
    fig.savefig(FIGDIR / f'{plot_stem}.svg', bbox_inches='tight')
    fig.savefig(FIGDIR / f'{plot_stem}.png', dpi=180, bbox_inches='tight')
    display(fig)
    plt.close(fig)
    display(pd.DataFrame([{'svg': str(FIGDIR / f'{plot_stem}.svg'), 'data': str(TABLEDIR / f'{plot_stem}_plot_data.csv'),
                           'unique_clones': len(frame), 'displayed_rows': len(plot_data), 'primary_contrast': str(example.contrast_label),
                           'primary_qvalue': q, 'q_family': q_family}]))
    '''
    for index, example_id in enumerate(['mono_1', 'mono_2', 'multi_divergence', 'multi_convergence'], 1):
        stem = f'program_{index:02d}_{example_id}_pooled_and_metaclone_distribution'
        source = distribution_source.replace('__EXAMPLE__', repr(example_id)).replace('__STEM__', repr(stem)).replace('__SEED__', str(index))
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    cells.append(nbf.v4.new_code_cell(textwrap.dedent(r'''
    # Chart contract: all fixed programs, population z-score of four fitted means.
    # An FDR symbol applies to a separately fitted MC-versus-peer contrast.
    plot_stem = 'program_05_monocyte_all50_row_zscore_heatmap'
    ordered_programs = sorted(PROGRAMS, key=lambda p: str(p).casefold())
    means = adjusted_means.loc[MONO_ORDER, ordered_programs].T.copy()
    center = means.mean(axis=1)
    scale = means.std(axis=1, ddof=0)
    z = means.sub(center, axis=0).div(scale.where(scale > 1e-12), axis=0)
    ovrs = primary[(primary.tier == 'within_lineage_ovr') & (primary.lineage_bias == 'Monocyte')].copy()
    ovrs['metaclone'] = ovrs.target_levels.str.replace('MC', '', regex=False).astype(int)
    ovrs = ovrs.set_index(['program', 'metaclone'])
    records = []
    for row_i, program in enumerate(ordered_programs):
        for column_i, mc in enumerate(MONO_ORDER):
            test = ovrs.loc[(program, mc)]
            value = float(z.loc[program, mc])
            records.append({'program': program, 'metaclone': mc, 'row_order': row_i, 'column_order': column_i,
                            'adjusted_program_mean_sd': float(means.loc[program, mc]), 'row_center': float(center.loc[program]),
                            'row_scale': float(scale.loc[program]), 'zscore_ddof': 0, 'raw_zscore': value,
                            'display_zscore': np.clip(value, -2.5, 2.5), 'was_clipped': bool(abs(value) > 2.5),
                            'zscore_estimable': bool(scale.loc[program] > 1e-12),
                            'undefined_reason': '' if scale.loc[program] > 1e-12 else 'constant adjusted means across four MCs',
                            'contrast_id': test.contrast_id, 'original_950_test_OVR_qvalue': float(test.qvalue),
                            'contrast_support_gate': bool(test.support_gate), 'effect_sd': float(test.effect_sd),
                            'symbol': 'circle' if test.support_gate and test.qvalue < 0.05 else 'square' if test.support_gate and test.qvalue < 0.10 else ''})
    plot_data = pd.DataFrame(records)
    plot_data.to_csv(TABLEDIR / f'{plot_stem}_plot_data.csv', index=False)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('program_all50_blue_white_red', ['#2166AC', '#FFFFFF', '#B2182B'], N=257)
    cmap.set_bad('#D9DEE5')
    norm = mpl.colors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    assert mpl.colors.to_hex(cmap(norm(0))) == '#ffffff'
    fig, ax = plt.subplots(figsize=(9.6, 15.0))
    fig.subplots_adjust(left=0.40, right=0.83, bottom=0.105, top=0.925)
    mesh = ax.pcolormesh(np.arange(5), np.arange(len(ordered_programs) + 1),
                         z.to_numpy(), cmap=cmap, norm=norm, shading='flat',
                         edgecolors='#EEF0F3', linewidth=0.40, rasterized=False)
    ax.set_ylim(len(ordered_programs), 0)
    ax.set_xticks(np.arange(4) + 0.5, [f'MC{mc}\n' + ('slower' if mc in [6, 8] else 'faster') for mc in MONO_ORDER], fontsize=10)
    ax.set_yticks(np.arange(len(ordered_programs)) + 0.5,
                  [str(p).replace('HALLMARK_', '').replace('_', ' ') for p in ordered_programs], fontsize=9.4)
    ax.tick_params(axis='both', length=0, pad=7)
    ax.axvline(2, color='#202833', linewidth=1.35)
    for symbol, marker in [('circle', 'o'), ('square', 's')]:
        chosen = plot_data[plot_data.symbol == symbol]
        ax.scatter(chosen.column_order + 0.5, chosen.row_order + 0.5,
                   marker=marker, s=24 if marker == 'o' else 22, facecolors='white',
                   edgecolors='#202833', linewidths=0.85, zorder=4)
    cax = fig.add_axes([0.86, 0.41, 0.020, 0.25])
    colorbar = fig.colorbar(mesh, cax=cax, ticks=[-2.5, 0, 2.5])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label('Within-program z-score', fontsize=10, labelpad=7)
    colorbar.ax.set_yticklabels(['−2.5', '0', '+2.5'])
    fig.text(0.08, 0.975, 'Monocyte program landscape: all 50 Hallmarks', ha='left', va='top',
             fontsize=17, weight='bold', color='#202833')
    fig.text(0.08, 0.947, 'Day-2 HSPCs · row z-scores of library-adjusted clone means across the four metaclones',
             ha='left', va='top', fontsize=10.4, color='#4D5868')
    legend_handles = [mpl.lines.Line2D([], [], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor='#202833', label='Original OVR q < 0.05'),
                      mpl.lines.Line2D([], [], marker='s', linestyle='none', markerfacecolor='white', markeredgecolor='#202833', label='Exploratory 0.05 ≤ q < 0.10')]
    fig.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0.08, 0.044), ncol=2, frameon=False, fontsize=10)
    fig.text(0.08, 0.035, 'Symbols: each MC versus equal-MC peers; original 950-test BH family. Color is not statistical support.',
             fontsize=9.3, ha='left', va='top', color='#4D5868')
    fig.text(0.08, 0.016, 'Population z-score (ddof=0); shared display limits ±2.5. Raw unclipped values and any undefined rows are retained in CSV.',
             fontsize=9.3, ha='left', va='top', color='#4D5868')
    fig.savefig(FIGDIR / f'{plot_stem}.svg', bbox_inches='tight')
    fig.savefig(FIGDIR / f'{plot_stem}.png', dpi=180, bbox_inches='tight')
    display(fig)
    plt.close(fig)
    display(pd.DataFrame([{'svg': str(FIGDIR / f'{plot_stem}.svg'), 'data': str(TABLEDIR / f'{plot_stem}_plot_data.csv'),
                           'programs': len(ordered_programs), 'tiles': len(plot_data), 'clipped_tiles': int(plot_data.was_clipped.sum()),
                           'undefined_tiles': int((~plot_data.zscore_estimable).sum())}]))
    ''').strip()))

    forest_source = r'''
    # Chart contract: one contrast across primary, source, and true cell-removal
    # refits; effects and pointwise intervals use frozen primary gene/program SDs.
    example_id = __EXAMPLE__
    plot_stem = __STEM__
    example = selection.loc[selection.example_id == example_id].iloc[0]
    program, contrast_id = str(example.program), str(example.contrast_id)
    primary_row = primary[(primary.program == program) & (primary.contrast_id == contrast_id)].iloc[0]
    rows = [{'variant': 'primary', 'variant_label': 'Primary pooled HSPCs', 'display_order': 0,
             'effect_primary_scale': float(primary_row.effect_sd), 'primary_scale_se': float(primary_row.standard_error),
             'support_gate': bool(primary_row.support_gate), 'estimable': bool(primary_row.estimable),
             'native_score_qvalue': float(primary_row.qvalue), 'q_scale': 'primary score scale',
             'shared_libraries': str(primary_row.shared_libraries)}]
    candidates = sensitivity[(sensitivity.program == program) & (sensitivity.contrast_id == contrast_id)].copy()
    normalization_variants = ['nonmito_poscounts'] if 'nonmito_poscounts' in set(candidates.variant) else []
    variants = ['source_fraction_adjusted', 'prog2_only'] + normalization_variants + sorted(candidates.loc[candidates.variant.str.startswith('without_library__'), 'variant'].unique())
    for position, variant in enumerate(variants, 1):
        r = candidates.loc[candidates.variant == variant].iloc[0]
        label = 'Adjust for prog_2 fraction' if variant == 'source_fraction_adjusted' else 'prog_2 cells only' if variant == 'prog2_only' else 'Non-mito poscounts normalization' if variant == 'nonmito_poscounts' else 'Leave out ' + (str(r.removed_library) if 'removed_library' in r and pd.notna(r.removed_library) and str(r.removed_library) else variant.replace('without_library__', ''))
        rows.append({'variant': variant, 'variant_label': label, 'display_order': position,
                     'effect_primary_scale': float(r.effect_primary_scale), 'primary_scale_se': float(r.primary_scale_se),
                     'support_gate': bool(r.support_gate), 'estimable': bool(r.estimable),
                     'native_score_qvalue': float(r.fit_tier_fdr), 'q_scale': 'native refit scale; not the plotted frozen-scale CI',
                     'shared_libraries': str(r.shared_libraries)})
    plot_data = pd.DataFrame(rows)
    plot_data['program'] = program
    plot_data['contrast_id'] = contrast_id
    plot_data['example_id'] = example_id
    plot_data['ci_low_primary_scale'] = plot_data.effect_primary_scale - 1.959963984540054 * plot_data.primary_scale_se
    plot_data['ci_high_primary_scale'] = plot_data.effect_primary_scale + 1.959963984540054 * plot_data.primary_scale_se
    plot_data['interval_type'] = 'pointwise 95% HC3 normal interval; frozen primary gene/program scale'
    plot_data['display_supported'] = plot_data.support_gate & plot_data.estimable & np.isfinite(plot_data.effect_primary_scale) & np.isfinite(plot_data.primary_scale_se)
    plot_data['undefined_reason'] = np.where(plot_data.display_supported, '', 'not estimable or insufficient shared-library support')
    plot_data.to_csv(TABLEDIR / f'{plot_stem}_plot_data.csv', index=False)

    fig, ax = plt.subplots(figsize=(11.3, 7.65 if normalization_variants else 7.2))
    fig.subplots_adjust(left=0.30, right=0.90, bottom=0.18, top=0.785)
    ax.axvline(0, color='#7E8998', linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.axhline(2.5 + len(normalization_variants), color='#D4D9E0', linewidth=0.9, zorder=1)
    for r in plot_data.itertuples(index=False):
        if r.display_supported:
            is_primary = r.variant == 'primary'
            color = '#B2182B' if is_primary else '#2166AC'
            ax.errorbar(r.effect_primary_scale, r.display_order,
                        xerr=1.959963984540054 * r.primary_scale_se, fmt='D' if is_primary else 'o',
                        color=color, markerfacecolor=color if not is_primary or primary_row.qvalue < .05 else 'white',
                        markersize=7 if is_primary else 5.5, capsize=3, elinewidth=1.1, markeredgewidth=1.1, zorder=3)
        else:
            ax.text(0.025, r.display_order, 'NE', transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=10, color='#677384', weight='bold')
    ax.set_yticks(plot_data.display_order, plot_data.variant_label, fontsize=10.3)
    ax.set_ylim(len(plot_data) - 0.45, -0.65)
    ax.set_xlabel('Program effect in frozen primary SD units\n(target minus reference; pointwise 95% HC3 interval)', fontsize=10.5)
    ax.grid(axis='x', color='#EBEEF2', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=9)
    title = program.replace('HALLMARK_', '').replace('_', ' ')
    fig.text(0.075, 0.956, title + ': sensitivity of the selected contrast', ha='left', va='top',
             fontsize=15.3, weight='bold', color='#202833')
    fig.text(0.075, 0.907, f'Day-2 clone summaries · {example.contrast_label}', ha='left', va='top', fontsize=11, color='#4D5868')
    q_family = '50-Hallmark pace BH' if example.context == 'Monocyte' else 'original 2,000-test pairwise BH'
    q = float(primary_row.qvalue)
    q_tier = 'q < 0.05' if q < .05 else 'exploratory 0.05 ≤ q < 0.10' if q < .10 else 'q ≥ 0.10'
    fig.text(0.075, 0.864, f'Primary {q_family}: q={q:.3g} ({q_tier}). Only the primary point carries this q annotation.',
             ha='left', va='top', fontsize=10.1, color='#283444')
    fig.text(0.075, 0.087, 'Library removal reaggregates remaining cells within each clone and refits normalization and the model.',
             ha='left', va='top', fontsize=9.4, color='#4D5868')
    fig.text(0.075, 0.052, 'NE: missing estimand or insufficient shared-library support. Refit q values use native scales and remain in CSV.',
             ha='left', va='top', fontsize=9.4, color='#4D5868')
    fig.savefig(FIGDIR / f'{plot_stem}.svg', bbox_inches='tight')
    fig.savefig(FIGDIR / f'{plot_stem}.png', dpi=180, bbox_inches='tight')
    display(fig)
    plt.close(fig)
    display(plot_data[['variant_label', 'effect_primary_scale', 'ci_low_primary_scale', 'ci_high_primary_scale', 'display_supported']])
    print(FIGDIR / f'{plot_stem}.svg')
    '''
    for index, example_id in [(6, 'mono_1'), (7, 'multi_divergence')]:
        stem = f'program_{index:02d}_{example_id}_sensitivity_forest'
        source = forest_source.replace('__EXAMPLE__', repr(example_id)).replace('__STEM__', repr(stem))
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    contribution_source = r'''
    # Chart contract: exact additive program-effect decomposition, top genes by
    # absolute contribution plus the signed sum of all remaining mapped genes.
    example_id = __EXAMPLE__
    plot_stem = __STEM__
    example = selection.loc[selection.example_id == example_id].iloc[0]
    gene_data = gene_contributions[gene_contributions.example_id == example_id].copy().sort_values('rank_abs_contribution')
    n_show = min(15, len(gene_data))
    shown = gene_data.head(n_show).copy()
    shown['display_label'] = shown.gene
    shown['display_role'] = 'individual gene'
    shown['n_genes_represented'] = 1
    if len(gene_data) > n_show:
        rest = gene_data.iloc[n_show:]
        other = {c: np.nan for c in shown.columns}
        other.update({'example_id': example_id, 'program': str(example.program), 'contrast_id': str(example.contrast_id),
                      'gene': '|'.join(rest.gene.astype(str)), 'rank_abs_contribution': n_show + 1,
                      'program_sd_contribution': float(rest.program_sd_contribution.sum()),
                      'program_effect_sd': float(example.effect_sd), 'program_mapped_genes': len(gene_data),
                      'selection': 'signed sum of all unshown mapped genes',
                      'display_label': f'Remaining {len(rest)} genes (sum)', 'display_role': 'remaining gene aggregate',
                      'n_genes_represented': len(rest)})
        shown = pd.concat([shown, pd.DataFrame([other])], ignore_index=True)
    plot_data = shown.reset_index(drop=True)
    plot_data['display_order'] = np.arange(len(plot_data))
    plot_data['contrast_label'] = str(example.contrast_label)
    plot_data['statistical_test'] = 'none; additive decomposition of fitted program effect'
    assert np.isclose(plot_data.program_sd_contribution.sum(), float(example.effect_sd), atol=1e-9)
    plot_data.to_csv(TABLEDIR / f'{plot_stem}_plot_data.csv', index=False)

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    fig.subplots_adjust(left=0.285, right=0.88, bottom=0.18, top=0.815)
    values = plot_data.program_sd_contribution.to_numpy(float)
    colors = ['#B2182B' if x >= 0 else '#2166AC' for x in values]
    bars = ax.barh(plot_data.display_order, values, height=0.62, color=colors, edgecolor=colors, linewidth=0.7)
    for bar, role in zip(bars, plot_data.display_role):
        if role == 'remaining gene aggregate': bar.set_facecolor('white'); bar.set_hatch('///')
    ax.axvline(0, color='#667283', linewidth=0.9)
    span = max(float(np.max(np.abs(values))), 0.01)
    ax.set_xlim(min(float(values.min()), 0) - 0.28 * span, max(float(values.max()), 0) + 0.31 * span)
    for r in plot_data.itertuples(index=False):
        sign = 1 if r.program_sd_contribution >= 0 else -1
        ax.text(r.program_sd_contribution + sign * 0.035 * span, r.display_order,
                f'{r.program_sd_contribution:+.3f}', ha='left' if sign > 0 else 'right',
                va='center', fontsize=9.2, color='#283444')
    ax.set_yticks(plot_data.display_order, plot_data.display_label, fontsize=10.1)
    ax.set_ylim(len(plot_data) - 0.3, -0.75)
    ax.set_xlabel('Signed contribution to the fitted program effect (SD units)', fontsize=10.5)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#EBEEF2', linewidth=0.65)
    ax.set_axisbelow(True)
    title = str(example.program).replace('HALLMARK_', '').replace('_', ' ')
    fig.text(0.075, 0.962, title + ': leading-gene contributions', ha='left', va='top',
             fontsize=16, weight='bold', color='#202833')
    fig.text(0.075, 0.916, f'Day-2 clone summaries · {example.contrast_label}', ha='left', va='top', fontsize=11, color='#4D5868')
    fig.text(0.075, 0.873, f'Top {n_show} genes by absolute contribution; all {len(gene_data)} mapped genes sum to {example.effect_sd:+.3f} program SD.',
             ha='left', va='top', fontsize=10.3, color='#283444')
    fig.text(0.075, 0.090, 'Contribution = adjusted gene-z contrast / (mapped-gene count × primary raw-program SD).',
             ha='left', va='top', fontsize=9.5, color='#4D5868')
    fig.text(0.075, 0.053, 'Bars are contributions, not confidence intervals or gene-level discoveries; hatched bar sums all remaining genes.',
             ha='left', va='top', fontsize=9.5, color='#4D5868')
    fig.savefig(FIGDIR / f'{plot_stem}.svg', bbox_inches='tight')
    fig.savefig(FIGDIR / f'{plot_stem}.png', dpi=180, bbox_inches='tight')
    display(fig)
    plt.close(fig)
    display(plot_data[['display_label', 'program_sd_contribution', 'n_genes_represented']])
    print(FIGDIR / f'{plot_stem}.svg')
    '''
    for index, example_id in [(8, 'mono_1'), (9, 'multi_divergence')]:
        stem = f'program_{index:02d}_{example_id}_leading_gene_contributions'
        source = contribution_source.replace('__EXAMPLE__', repr(example_id)).replace('__STEM__', repr(stem))
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    return cells
