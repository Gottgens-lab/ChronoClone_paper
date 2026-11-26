from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parent
cells = []
cells.append(nbf.v4.new_markdown_cell('''# Singh Day-7 HSPC program annotation from retained gene content

Gene-grounded descriptive annotation of the existing RNA-only eight-program NMF; no model refit, new differential-expression analysis, hypothesis test, enrichment rerun, or program selection. All eight retained programs are inspected. Program names are descriptive interpretations, not cell identities or causal mechanisms.

Environment: **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`; kernel **chronoclone**.

Execution command:
`/path/to/ChronoClone/.runtime/chronoclone/bin/python /path/to/ChronoClone/analysis_workspace/singh/singh_weinreb_style_umaps_20260911/execute_notebook.py /path/to/ChronoClone/analysis_workspace/singh/singh_program_annotation_20260911/0001_gene_grounded_program_annotations.ipynb`
'''))
cells.append(nbf.v4.new_code_cell('''from pathlib import Path
import hashlib, json, sys, platform, importlib.metadata
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
BASE = Path('/path/to/ChronoClone/analysis_workspace/singh/singh_program_annotation_20260911')
SOURCE = Path('/path/to/ChronoClone/analysis_workspace/singh/singh_reference_response_20260911/programs')
for sub in ['tables', 'reports', 'panels']:
    (BASE / sub).mkdir(exist_ok=True)
environment = {'name': 'chronoclone', 'path': '/path/to/ChronoClone/.runtime/chronoclone', 'kernel': 'chronoclone', 'executable': sys.executable, 'python': platform.python_version(), 'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','nbformat','nbclient']}}
assert Path(sys.executable).parent.parent == Path(environment['path'])
(BASE / 'reports/environment.json').write_text(json.dumps(environment, indent=2))
display(environment)
inputs = [SOURCE / 'tables' / f for f in ['program_spectra.csv','program_query_genes.csv','program_identity_annotations.csv','program_enrichment_complete_family.csv','metaclone_mean_usage.csv','metaclone_program_conditional_intervals.csv','retrospective_mc3_mc7_characterization.csv','metaclone_depth.csv','metaclone_relative_usage_zscore.csv','06c_profile_plot_values.csv']]
inputs.append(SOURCE / '0002_rna_normalized_day7_programs.ipynb')
source_manifest = pd.DataFrame([{'source': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in inputs])
source_manifest.to_csv(BASE / 'reports/source_manifest.csv', index=False)
display(source_manifest)
spectra = pd.read_csv(SOURCE / 'tables/program_spectra.csv', index_col=0)
query_genes = pd.read_csv(SOURCE / 'tables/program_query_genes.csv')
old_annotations = pd.read_csv(SOURCE / 'tables/program_identity_annotations.csv')
enrichment = pd.read_csv(SOURCE / 'tables/program_enrichment_complete_family.csv')
comparison = pd.read_csv(SOURCE / 'tables/retrospective_mc3_mc7_characterization.csv')
assert spectra.shape == (8, 2000)
assert list(spectra.index) == [f'P{i}' for i in range(1,9)]
assert not spectra.columns.duplicated().any()
assert len(enrichment) == 30624
Hn = spectra.div(spectra.sum(axis=1), axis=0)
specificity = Hn.pow(2).div(Hn.sum(axis=0), axis=1).fillna(0)
gene_rows = []
for program in spectra.index:
    loading_rank = spectra.loc[program].rank(method='first', ascending=False).astype(int)
    specificity_rank = specificity.loc[program].rank(method='first', ascending=False).astype(int)
    gene_rows.append(pd.DataFrame({'program': program, 'gene': spectra.columns, 'loading': spectra.loc[program].values, 'normalized_loading': Hn.loc[program].values, 'loading_rank': loading_rank.values, 'specificity': specificity.loc[program].values, 'specificity_rank': specificity_rank.values}))
gene_evidence = pd.concat(gene_rows, ignore_index=True)
gene_evidence.to_csv(BASE / 'tables/all_program_gene_evidence.csv', index=False)
# Verify the retained top-100 lists are reproduced; there is no new enrichment test.
for program in spectra.index:
    for query, rank_column in [('top_loading','loading_rank'), ('specificity','specificity_rank')]:
        available_queries = set(query_genes['query'])
        actual_query = query if query in available_queries else 'specific'
        saved = query_genes.loc[(query_genes.program == program) & (query_genes['query'] == actual_query)].sort_values('rank')['gene'].tolist()
        current = gene_evidence.loc[gene_evidence.program == program].sort_values(rank_column).head(100)['gene'].tolist()
        assert current == saved, (program, actual_query)
display({'spectra_shape': spectra.shape, 'gene_evidence_rows': len(gene_evidence), 'saved_query_types': query_genes['query'].unique().tolist(), 'complete_existing_enrichment_tests': len(enrichment)})
display(comparison)
'''))
for program in range(1,9):
    cells.append(nbf.v4.new_code_cell(f'''# P{program}: inspect both absolute spectrum weights and specificity, not only the selected ontology label.
program = 'P{program}'
rows = gene_evidence.loc[gene_evidence.program == program]
display(Markdown('## ' + program))
display(rows.sort_values('specificity_rank').head(40).reset_index(drop=True))
display(Markdown('Highest raw-loading genes'))
display(rows.sort_values('loading_rank').head(25).reset_index(drop=True))
display(old_annotations.loc[old_annotations.program == program])
print('Specificity top 40:', ', '.join(rows.sort_values('specificity_rank').head(40).gene))
print('Loading top 25:', ', '.join(rows.sort_values('loading_rank').head(25).gene))
'''))
cells.append(nbf.v4.new_markdown_cell('''## Literature-informed curation

The records below are human-readable interpretations of the retained gene weights, not newly fitted classes or statistically selected metaclone signatures. Literature supports roles of individual genes in the stated experimental contexts; it does not establish that these NMF programs cause the observed clonal histories. In particular, HSPC is the pooled input annotation throughout. No HSC/MPP subdivision is reconstructed. Three representative genes per program are curated for the loading display from the first 40 loading or specificity ranks; they are not automatically the three highest genes.
'''))
cells.append(nbf.v4.new_code_cell('''# Descriptive curation registry: interpretation is explicit and separate from measured weights.
annotation_records = [
    {'program':'P1', 'label':'HSC-associated regulation', 'plot_label':'HSC-associated regulation', 'anchors':['Hlf','Mecom','Procr'], 'additional_genes':['Ly6a','Prdm16','Rora','Sox4'], 'interpretation':'Hlf, Mecom, Procr and Ly6a support an HSC-associated regulatory/marker mixture within the pooled HSPC compartment.', 'boundary':'Not a purified HSC identity, measured quiescence, self-renewal assay, or a new HSC/MPP subdivision. Adgrl4 and Nrxn1 contribute strongly and the spectrum is mixed.', 'references':['Komorowska2017','Yokomizo2019']},
    {'program':'P2', 'label':'Biosynthesis / metabolism', 'plot_label':'Biosynthesis / metabolism', 'anchors':['Nop10','Fdps','Ndufa4'], 'additional_genes':['Fabp5','Dbi','Nme1','Cycs','Srm','Rpl11','Rps15a'], 'interpretation':'Ribosomal/RNA-processing genes occur together with lipid/isoprenoid and respiratory-metabolic machinery; this is a broad biosynthetic/metabolic component.', 'boundary':'Not a lineage-specific program or evidence of greater translation, respiration, or metabolic flux. Ndufa4 is assigned to respiratory complex IV, not complex I.', 'references':['Khanna2006','Balsa2012']},
    {'program':'P3', 'label':'G2/M mitosis', 'plot_label':'G2/M mitosis', 'anchors':['Ube2c','Ccnb1','Cenpf'], 'additional_genes':['Nusap1','Cenpa','Tpx2','Birc5','Cdc20','Plk1','Aurka'], 'interpretation':'A coherent mitotic/spindle/chromosome-segregation component, consistent with the existing G2-M Checkpoint annotation.', 'boundary':'Program usage is not a measured division rate and does not identify lineage output or self-renewal.', 'references':['Passegue2005']},
    {'program':'P4', 'label':'Integrated stress response', 'plot_label':'Integrated stress response', 'anchors':['Atf4','Trib3','Asns'], 'additional_genes':['Atf5','Nupr1','Ddit3','Chac1','Slc7a11','Ppp1r15a','Herpud1'], 'interpretation':'ATF4-associated amino-acid, redox and proteostasis response genes provide a more specific description than the generic cellular-stimuli term.', 'boundary':'A transcriptional stress-associated mixture, not a direct assay of eIF2alpha phosphorylation, ATF4 protein activity, stress cause, injury, or inferior stem-cell function; ISR can support HSC persistence.', 'references':['vanGalen2018']},
    {'program':'P5', 'label':'CDK6 / immune-associated', 'plot_label':'CDK6 / immune-associated', 'anchors':['Cdk6','Cd48','Nkg7'], 'additional_genes':['Tespa1','Lat2','Fcer1g','Clec4e','Cpa3','Gzmb','Itga2b'], 'interpretation':'Cdk6 and Cd48 place the component in a progenitor-activation context, accompanied by immune-signaling/granule genes spanning more than one mature lineage.', 'boundary':'Mixed program: not platelet identity, NK-cell identity, proven quiescence exit, or a specific commitment route. Cdk6 RNA does not establish CDK6 protein or kinase activity.', 'references':['Laurenti2015','Boles2011','Ng2020','Wang2014']},
    {'program':'P6', 'label':'DNA replication / repair', 'plot_label':'DNA replication / repair', 'anchors':['Pola1','Mcm4','Rad51b'], 'additional_genes':['Pcna','Tk1','Rrm2','Lig1','Rad51ap1','Hells','Gmnn','Tyms'], 'interpretation':'Replication licensing/synthesis and homologous-recombination-associated genes distinguish this component from the predominantly mitotic P3 program; consistent with existing E2F Targets annotation.', 'boundary':'Not evidence of DNA damage, mutation burden, measured repair activity, or greater cell-division rate.', 'references':['Passegue2005']},
    {'program':'P7', 'label':'Glycolysis / hypoxia-associated', 'plot_label':'Glycolysis / hypoxia-associated', 'anchors':['Pgk1','Bnip3','Egln3'], 'additional_genes':['Eno1','Tpi1','Aldoa','Aldoc','Bhlhe40','Ankrd37','Csf3r','Prtn3'], 'interpretation':'A coherent glycolytic and hypoxia-response-associated core occurs with immediate-early and some myeloid-associated genes.', 'boundary':'Not measured oxygen tension, glycolytic flux, or a causal driver of faster differentiation. Glycolysis can also support HSC quiescence; the myeloid genes do not establish a pure granulocytic program.', 'references':['Takubo2013']},
    {'program':'P8', 'label':'HSPC regulation / adhesion', 'plot_label':'HSPC regulation / adhesion', 'anchors':['Pbx1','Zeb2','Tgfbr3'], 'additional_genes':['Med12l','Runx1','Mecom','Ncam2','Enah','Itga6','Slamf1','Palld'], 'interpretation':'Hematopoietic regulatory genes coexist with receptors, adhesion and cytoskeletal genes; this provides an HSPC-relevant description of a mixed component.', 'boundary':'Not neuronal fate, a pure HSC identity, proven TGF-beta pathway activity, measured adhesion, or self-renewal. MED12L is not MED12; Ncam2 alone does not establish a hematopoietic mechanism.', 'references':['Ficara2008','Goossens2011']},
]
references = [
 {'key':'Komorowska2017','citation':'Komorowska, K., Doyle, A., Wahlestedt, M., et al. (2017). Hepatic Leukemia Factor Maintains Quiescence of Hematopoietic Stem Cells and Protects the Stem Cell Pool during Regeneration. Cell Reports 21, 3514–3523.','doi':'10.1016/j.celrep.2017.11.084','url':'https://pubmed.ncbi.nlm.nih.gov/29262330/','support':'Mouse Hlf deletion impaired regenerative HSC maintenance and reduced quiescence. This supports Hlf-associated context, not quiescence measurements in Singh cells.'},
 {'key':'Yokomizo2019','citation':'Yokomizo, T., Watanabe, N., Umemoto, T., et al. (2019). Hlf marks the developmental pathway for hematopoietic stem cells but not for erythro-myeloid progenitors. Journal of Experimental Medicine 216, 1599–1614.','doi':'10.1084/jem.20181399','url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6605751/','support':'Developmental mouse Hlf-positive cells contained Hlf, Mecom and Procr-associated HSC expression; developmental setting, not an adult lineage identity classifier.'},
 {'key':'Khanna2006','citation':'Khanna, M., Wu, H., Johansson, C., Caizergues-Ferrer, M., and Feigon, J. (2006). Structural study of the H/ACA snoRNP components Nop10p and the 3-prime hairpin of U65 snoRNA. RNA 12, 40–52.','doi':'10.1261/rna.2221606','url':'https://pubmed.ncbi.nlm.nih.gov/16373493/','support':'Nop10-related snoRNP structure and RNA modification. Molecular-function support, not hematopoietic-state evidence.'},
 {'key':'Balsa2012','citation':'Balsa, E., Marco, R., Perales-Clemente, E., Szklarczyk, R., Calvo, E., Landazuri, M.O., and Enriquez, J.A. (2012). NDUFA4 is a subunit of complex IV of the mammalian electron transport chain. Cell Metabolism 16, 378–386.','doi':'10.1016/j.cmet.2012.07.015','url':'https://pubmed.ncbi.nlm.nih.gov/22902835/','support':'NDUFA4 respiratory-complex-IV function; does not measure metabolic flux here.'},
 {'key':'Passegue2005','citation':'Passegue, E., Wagers, A.J., Giuriato, S., Anderson, W.C., and Weissman, I.L. (2005). Global analysis of proliferation and cell cycle gene expression in the regulation of hematopoietic stem and progenitor cell fates. Journal of Experimental Medicine 202, 1599–1611.','doi':'10.1084/jem.20050967','url':'https://pubmed.ncbi.nlm.nih.gov/16330818/','support':'Primary HSPC cell-cycle and proliferation study; contextualizes phase-associated expression without equating NMF usage with division rate.'},
 {'key':'vanGalen2018','citation':'van Galen, P., Mbong, N., Kreso, A., Schoof, E.M., Wagenblast, E., Ng, S.W.K., Krivdova, G., Jin, L., Nakauchi, H., and Dick, J.E. (2018). Integrated Stress Response Activity Marks Stem Cells in Normal Hematopoiesis and Leukemia. Cell Reports 25, 1109–1117.e5.','doi':'10.1016/j.celrep.2018.10.021','url':'https://www.sciencedirect.com/science/article/pii/S2211124718315973','support':'Primary human HSC study links ATF4/ISR and amino-acid deprivation response with persistence. Prevents interpreting stress-associated expression as loss of stem function.'},
 {'key':'Laurenti2015','citation':'Laurenti, E., Frelin, C., Xie, S., et al. (2015). CDK6 levels regulate quiescence exit in human hematopoietic stem cells. Cell Stem Cell 16, 302–313.','doi':'10.1016/j.stem.2015.01.017','url':'https://pubmed.ncbi.nlm.nih.gov/25704240/','support':'CDK6 protein governs quiescence-exit kinetics in purified human HSC subsets. This motivates activation-associated interpretation, not proof from RNA or a new HSC subdivision.'},
 {'key':'Boles2011','citation':'Boles, N.C., Lin, K.K., Lukov, G.L., Bowman, T.V., Baldridge, M.T., and Goodell, M.A. (2011). CD48 on hematopoietic progenitors regulates stem cells and suppresses tumor formation. Blood 118, 80–87.','doi':'10.1182/blood-2010-12-322339','url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC3139389/','support':'CD48 protein was present on short-term progenitors and excluded from quiescent LT-HSCs in the examined mouse setting. Not a direct classification from Cd48 RNA here.'},
 {'key':'Ng2020','citation':'Ng, S.S., De Labastida Rivera, F., Yan, J., et al. (2020). The NK cell granule protein NKG7 regulates cytotoxic granule exocytosis and inflammation. Nature Immunology 21, 1205–1218.','doi':'10.1038/s41590-020-0758-6','url':'https://pubmed.ncbi.nlm.nih.gov/32839608/','support':'NKG7 granule/immune-effector function; its presence alongside heterogeneous genes does not establish NK identity.'},
 {'key':'Wang2014','citation':'Wang, D., Zheng, M., Qiu, Y., et al. (2014). Tespa1 negatively regulates Fc-epsilon-RI-mediated signaling and the mast cell-mediated allergic response. Journal of Experimental Medicine.','doi':'10.1084/jem.20140470','url':'https://pubmed.ncbi.nlm.nih.gov/25422497/','support':'Tespa1 participates in mast-cell receptor signaling, demonstrating that this P5 anchor is not a platelet-specific marker.'},
 {'key':'Takubo2013','citation':'Takubo, K., Nagamatsu, G., Kobayashi, C.I., et al. (2013). Regulation of glycolysis by Pdk functions as a metabolic checkpoint for cell cycle quiescence in hematopoietic stem cells. Cell Stem Cell 12, 49–61.','doi':'10.1016/j.stem.2012.10.011','url':'https://pubmed.ncbi.nlm.nih.gov/23290136/','support':'Functional mouse HSC evidence that glycolysis can support quiescence; forbids treating a glycolytic RNA program as a universal differentiation-speed marker.'},
 {'key':'Ficara2008','citation':'Ficara, F., Murphy, M.J., Lin, M., and Cleary, M.L. (2008). Pbx1 regulates self-renewal of long-term hematopoietic stem cells by maintaining their quiescence. Cell Stem Cell 2, 484–496.','doi':'10.1016/j.stem.2008.03.004','url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC2416441/','support':'Conditional mouse Pbx1 deletion altered HSC maintenance and TGF-beta responses; context for a loaded regulator, not proof of the corresponding function in MC3.'},
 {'key':'Goossens2011','citation':'Goossens, S., Janzen, V., Bartunkova, S., et al. (2011). The EMT regulator Zeb2/Sip1 is essential for murine embryonic hematopoietic stem/progenitor cell differentiation and mobilization. Blood.','doi':'10.1182/blood-2010-08-300236','url':'https://pubmed.ncbi.nlm.nih.gov/21355089/','support':'Zeb2 deletion alters embryonic HSPC differentiation, mobilization and adhesion. Supports hematopoietic relevance rather than the misleading neuronal label; does not imply the same embryonic mechanism here.'},
]
annotations = pd.DataFrame(annotation_records).set_index('program')
assert list(annotations.index) == list(spectra.index)
assert set(sum(annotations.references.tolist(), [])) <= {r['key'] for r in references}
curated_genes = pd.DataFrame([{'program': p, 'gene': g, 'display_order': j + 1} for p, row in annotations.iterrows() for j, g in enumerate(row.anchors)])
curated_genes = curated_genes.merge(gene_evidence, on=['program','gene'], validate='one_to_one')
assert len(curated_genes) == 24
assert ((curated_genes.loading_rank <= 40) | (curated_genes.specificity_rank <= 40)).all()
curated_genes.to_csv(BASE / 'tables/representative_gene_display.csv', index=False)
annotations.reset_index().to_json(BASE / 'tables/program_annotations.json', orient='records', indent=2)
annotations.map(lambda x: '; '.join(x) if isinstance(x, list) else x).to_csv(BASE / 'tables/program_annotations.csv')
pd.DataFrame(references).to_csv(BASE / 'tables/primary_literature.csv', index=False)
display(annotations[['label','anchors','interpretation','boundary']])
display(curated_genes[['program','gene','loading_rank','specificity_rank']])
display(pd.DataFrame(references))
# Preserve all pre-existing annotations and q-values as source evidence; no significance is attached to the new descriptive labels.
old_annotations.to_csv(BASE / 'tables/retained_original_enrichment_labels.csv', index=False)
annotated_comparison = comparison.merge(annotations[['label']].reset_index(), on='program', validate='one_to_one')
annotated_comparison.to_csv(BASE / 'tables/annotated_mc3_mc7_comparison.csv', index=False)
display(annotated_comparison[['program','label','MC3','MC7','MC3_minus_MC7']])
print('The retained program with the largest absolute MC3-minus-MC7 difference is:', annotated_comparison.loc[annotated_comparison.MC3_minus_MC7.abs().idxmax(), 'program'])
'''))
cells.append(nbf.v4.new_code_cell('''# One raw plot: same saved all-metaclone z-scores; labels use the locked metaclone palette.
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.family':'Arial','font.size':16,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
mc_colors = {0:'#f77189',1:'#f8763b',2:'#d28e32',3:'#b79932',4:'#9ea231',5:'#7caa31',6:'#31b345',7:'#34b081',9:'#35ae9a',10:'#36acad',11:'#38aac1',12:'#3aa6dd',13:'#7699f4',15:'#b087f4',16:'#df6df4',17:'#f563d8',19:'#f66ab2'}
profiles_z = pd.read_csv(SOURCE / 'tables/metaclone_relative_usage_zscore.csv', index_col=0)
depth = pd.read_csv(SOURCE / 'tables/metaclone_depth.csv', index_col=0)
mc_order = [9,12,7,5,11,1,10,6,16,17,19,0,3,2,15,4,13]
program_order = ['P4','P5','P7','P1','P8','P3','P2','P6']
plot_data = profiles_z.loc[mc_order, program_order]
plot_data.to_csv(BASE / 'tables/07a_plot_values.csv')
fig, ax = plt.subplots(figsize=(9.4,9.2))
limit = max(2., float(np.ceil(np.abs(plot_data.to_numpy()).max()*2)/2))
im = ax.pcolormesh(plot_data.to_numpy(), cmap='RdBu_r', vmin=-limit, vmax=limit, shading='flat', rasterized=False)
ax.invert_yaxis()
ax.set_xticks(np.arange(8)+.5, program_order)
ax.set_yticks(np.arange(len(mc_order))+.5, [f'MC{m}  (n={depth.loc[m,"n_families"]})' for m in mc_order])
for tick, mc in zip(ax.get_yticklabels(), mc_order):
    tick.set_color(mc_colors[mc])
ax.set_xlabel('Day-7 HSPC program'); ax.set_ylabel('Metaclone (families)')
ax.tick_params(length=0,pad=9)
ax.spines[['top','right','bottom','left']].set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.045)
cbar.solids.set_rasterized(False)
cbar.set_label('Relative program usage (z-score)', labelpad=12)
fig.tight_layout()
stem = BASE / 'panels/07a_day7_metaclone_program_usage'
fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
plt.show()
display({'output':str(stem), 'unchanged_saved_values':True, 'zscore_limit':limit})
'''))
cells.append(nbf.v4.new_code_cell('''# One raw plot: three literature-interpretable genes per program, from top-40 loading or specificity ranks.
program_order = list(spectra.index)
selected_genes = curated_genes.sort_values(['program','display_order']).gene.tolist()
assert len(set(selected_genes)) == len(selected_genes)
normalized_loadings = spectra.div(spectra.sum(axis=1), axis=0)
load_matrix = normalized_loadings.loc[program_order, selected_genes].T
relative_loading = load_matrix.div(load_matrix.max(axis=1), axis=0)
relative_loading.to_csv(BASE / 'tables/07b_plot_values.csv')
fig, ax = plt.subplots(figsize=(7.8,10.0))
im = ax.pcolormesh(relative_loading.to_numpy(), cmap='Blues', vmin=0, vmax=1, shading='flat', rasterized=False)
ax.invert_yaxis()
ax.set_xticks(np.arange(8)+.5, program_order)
ax.set_yticks(np.arange(len(selected_genes))+.5, selected_genes, fontstyle='italic')
ax.set_xlabel('Day-7 HSPC program'); ax.set_ylabel('Representative program gene')
ax.tick_params(length=0,pad=8)
ax.spines[['top','right','bottom','left']].set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.055)
cbar.solids.set_rasterized(False)
cbar.set_label('Relative loading (gene maximum = 1)', labelpad=12)
fig.tight_layout()
stem = BASE / 'panels/07b_day7_representative_gene_loadings'
fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
plt.show()
display({'output':str(stem), 'representative_gene_selection':'Manual biological curation within top-40 loading or specificity ranks; not highest-three selection', 'saved_spectra_unchanged':True})
'''))
cells.append(nbf.v4.new_code_cell('''# One raw plot: retain every existing MC3/MC7 mean and interval exactly; annotate all eight programs.
plot_table = pd.read_csv(SOURCE / 'tables/06c_profile_plot_values.csv')
program_order = list(spectra.index)
assert plot_table.shape[0] == 16
assert set(plot_table.mc) == {3,7}
plot_table.to_csv(BASE / 'tables/07c_plot_values.csv', index=False)
pd.testing.assert_frame_equal(plot_table, pd.read_csv(BASE / 'tables/07c_plot_values.csv'), check_exact=True)
fig, ax = plt.subplots(figsize=(12.8,7.8))
styles = {3:(mc_colors[3],'o',-.15), 7:(mc_colors[7],'s',.15)}
for mc, (color, marker, offset) in styles.items():
    d = plot_table.loc[plot_table.mc == mc].set_index('program').loc[program_order]
    y = np.arange(8) + offset
    ax.errorbar(100*d['mean'], y, xerr=100*np.vstack([d['mean']-d.lower, d.upper-d['mean']]), fmt=marker, color=color, mec=color, mfc=color if mc == 3 else 'white', markersize=10, elinewidth=2.1, capsize=4.5, label=f'MC{mc} (n={int(d.n_families.iloc[0])})')
ax.set_yticks(np.arange(8), [f'{p}  {annotations.loc[p,"plot_label"]}' for p in program_order])
ax.invert_yaxis(); ax.set_xlim(left=0)
ax.set_xlabel('Mean program usage (%)'); ax.set_ylabel('')
ax.tick_params(axis='y', length=0, pad=12)
ax.grid(axis='x',color='#e6e6e6',linewidth=.7); ax.set_axisbelow(True)
legend = ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5,1.015), ncol=2, fontsize=16)
for text, mc in zip(legend.get_texts(), [3,7]):
    text.set_color(mc_colors[mc])
fig.tight_layout()
stem = BASE / 'panels/07c_day7_mc3_mc7_annotated_profiles'
fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
plt.show()
display({'output':str(stem), 'all_original_means_and_intervals_retained':True, 'intervals':'Existing pointwise 2.5th/97.5th percentiles of 2000 within-metaclone family resamples with NMF and partition fixed; not biological-replicate or contrast confidence intervals'})
'''))
cells.append(nbf.v4.new_code_cell('''# Manuscript integration packet computed from the unchanged saved estimates.
import re
reference_lookup = {}
for r in references:
    matched = re.match(r'(.*?) \\((\\d{4})\\)\\. (.*)', r['citation'])
    authors, year, remaining = matched.groups()
    title = remaining.split('. ')[0]
    reference_lookup[r['key']] = {'authors':authors, 'year':int(year), 'title':title, 'doi':r['doi'], 'url':r['url'], 'citation':r['citation'], 'gene_level_support':r['support']}
numeric_profiles = pd.read_csv(SOURCE / 'tables/06c_profile_plot_values.csv')
means = numeric_profiles.pivot(index='program', columns='mc', values='mean') * 100
paragraphs = [
    'Gene-level inspection placed the eight programs in a hematopoietic context (Figure 7B). P1 contained Hlf, Mecom and Procr, supporting an HSC-associated regulatory annotation [Komorowska2017; Yokomizo2019]. P8 combined Pbx1, Zeb2 and Runx1 with receptor, adhesion and cytoskeletal genes, including Tgfbr3, Ncam2 and Itga6 [Ficara2008; Goossens2011]. P4 contained the ATF4-associated stress-response genes Trib3, Asns and Ddit3; P7 combined glycolytic enzymes with Bnip3 and Egln3. P5 contained Cdk6 and Cd48 together with immune-associated genes, including Nkg7, Tespa1 and Fcer1g, rather than a coherent platelet-specific identity. P2 encompassed broad biosynthetic/metabolic machinery, whereas P3 and P6 separated mitotic from DNA-replication/repair-associated expression.',
    f'These molecular mixtures distinguished the early relatives of the MC3 and MC7 reference histories (Figure 7C). The largest observed difference was in the mixed CDK6/immune-associated program P5, with mean fractional usage of {means.loc["P5",3]:.1f}% in MC3 and {means.loc["P5",7]:.1f}% in MC7. The Cdk6/Cd48 content is compatible with a progenitor-activation context [Laurenti2015; Boles2011], without establishing mature immune identity. MC3 instead showed higher usage of P1 ({means.loc["P1",3]:.1f}% versus {means.loc["P1",7]:.1f}%) and P8 ({means.loc["P8",3]:.1f}% versus {means.loc["P8",7]:.1f}%). MC7 had higher stress-associated P4 ({means.loc["P4",7]:.1f}% versus {means.loc["P4",3]:.1f}%) and glycolysis/hypoxia-associated P7 ({means.loc["P7",7]:.1f}% versus {means.loc["P7",3]:.1f}%) usage, while the mitotic and replication-associated profiles were more similar.',
    'Thus, metaclones connect distinct reference histories to overlapping early regulatory, activation-associated and metabolic programs. These retrospective descriptions do not establish functional stem-cell identity, a causal differentiation mechanism or prediction of mutation response. In particular, stress responses and glycolysis can support HSC maintenance, so neither is a unidirectional marker of differentiation [vanGalen2018; Takubo2013].',
]
selection = 'Three representative genes per program were selected for biological interpretability from the top 40 raw-loading or specificity ranks, independently of MC3–MC7 effect magnitude. They are not necessarily the three highest-ranking genes or exclusive to one program. All 16,000 gene weights, all ranks, and the complete pre-existing enrichment family remain available.'
panel_paths = {'A':'panels/07a_day7_metaclone_program_usage.png','B':'panels/07b_day7_representative_gene_loadings.png','C':'panels/07c_day7_mc3_mc7_annotated_profiles.png'}
packet = {'programs':[{'program':p,'short_label':r.label,'anchor_genes':r.anchors,'additional_genes':r.additional_genes,'interpretation':r.interpretation,'caveat':r.boundary,'primary_references':[reference_lookup[k] for k in r.references]} for p,r in annotations.iterrows()], 'mc3_mc7_profiles':numeric_profiles.to_dict(orient='records'), 'proposed_results_paragraphs':paragraphs, 'figure7b_selection_description':selection, 'panels':panel_paths, 'interval_definition':'Existing pointwise 2.5th/97.5th percentiles of 2000 within-metaclone family resamples; fixed NMF and fixed partition. No new bootstrap, contrast test, or biological-replicate interval.', 'source_model':'analysis_workspace/singh/singh_reference_response_20260911/programs/0002_rna_normalized_day7_programs.ipynb', 'environment':environment, 'references':reference_lookup}
(BASE / 'reports/manuscript_annotation.json').write_text(json.dumps(packet, indent=2))
report = ['# Singh program annotations for manuscript integration', '', 'Gene-based descriptive annotation only. Existing chronoclone environment: `/path/to/ChronoClone/.runtime/chronoclone`; kernel `chronoclone`.', '', '## Recommended biological reading', '']
report.extend(paragraphs)
report.extend(['', '## All-eight annotation registry', '', '| Program | Descriptive label | Representative genes | Interpretation boundary |', '|---|---|---|---|'])
for p,r in annotations.iterrows():
    report.append(f'| {p} | {r.label} | {", ".join(r.anchors)} | {r.boundary} |')
report.extend(['', '## Figure captions and selection', '', '**7A.** Family-balanced relative program usage across all 17 reference metaclones. Values are the unchanged saved z-scores, centered/scaled using the previously declared metaclones with at least 32 Day-7 cells. Parenthetical n denotes resolved Day-7 families. No metaclone-specific significance tests.', '', '**7B.** Relative spectrum loadings for representative program-associated genes. Each program spectrum is normalized to sum to one and each displayed gene is scaled to its maximum across programs. ' + selection, '', '**7C.** Mean fractional program usage in MC3 and MC7 Day-7 HSPCs, with descriptive gene-grounded program labels. Means and intervals are copied unchanged from the retained analysis. ' + packet['interval_definition'] + ' Both source populations remain pooled HSPCs; these labels do not subdivide or reassign cells.', '', '## Important boundaries', '', '- Program labels are supported by gene content and literature, not new statistically significant MC identities. The original selected ontology terms and q-values remain in a transparency table; those q-values do not quantify confidence in the new manual names.', '- P5 is the largest observed MC3–MC7 program-usage difference and is explicitly included. Its mixed immune genes span several mature contexts, excluding a justified platelet-only, NK-only or mast-only call.', '- P8 is not a neuronal program. MED12L should not inherit functional claims from its distinct paralog MED12.', '- P7 includes Gt(ROSA)26Sor among its retained high-specificity features; this locus is not used as a biological annotation anchor. The saved fit and original RNA eligibility remain unchanged in this annotation-only task.', '- No causal interpretation, prospective prediction, sex/replicate modeling, HSC/MPP split, new DE, enrichment rerun, NMF refit, or new significance threshold.', '', '## Primary references', ''])
for k,r in reference_lookup.items():
    report.append(f'**{k}** {r["citation"]} DOI: [{r["doi"]}](https://doi.org/{r["doi"]}). {r["gene_level_support"]}')
(BASE / 'PROGRAM_ANNOTATION_HANDOFF.md').write_text('\\n\\n'.join(report))
display(Markdown('\\n\\n'.join(paragraphs)))
display({'integration_packet':str(BASE / 'reports/manuscript_annotation.json'), 'raw_panels':panel_paths, 'proposed_prose_word_count':sum(len(p.split()) for p in paragraphs)})
'''))
cells.append(nbf.v4.new_code_cell('''# Source preservation and output integrity; no new statistical tests.
import xml.etree.ElementTree as ET
for row in source_manifest.itertuples():
    assert hashlib.sha256(Path(row.source).read_bytes()).hexdigest() == row.sha256
checks=[]
for svg in sorted((BASE / 'panels').glob('*.svg')):
    elements=list(ET.parse(svg).getroot().iter())
    texts=[e for e in elements if e.tag.endswith('}text')]
    images=[e for e in elements if e.tag.endswith('}image')]
    checks.append({'panel':svg.stem,'editable_text_elements':len(texts),'raster_images':len(images),'png_exists':svg.with_suffix('.png').exists()})
    assert len(texts)>0 and len(images)==0 and svg.with_suffix('.png').exists()
pd.DataFrame(checks).to_csv(BASE / 'reports/svg_checks.csv', index=False)
outputs=[p for sub in ['tables','panels'] for p in (BASE/sub).glob('*') if p.is_file()]
pd.DataFrame([{'path':str(p.relative_to(BASE)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(outputs)]).to_csv(BASE / 'reports/output_manifest.csv', index=False)
display(pd.DataFrame(checks))
display({'source_hashes_unchanged':len(source_manifest), 'all_programs_descriptively_annotated':len(annotations), 'retained_NMF_model':'ordinary fixed-K8 NMF', 'new_model_fits':0, 'new_hypothesis_tests':0, 'program_and_metaclone_counts_unchanged':True})
'''))
book = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python (chronoclone)', 'language':'python', 'name':'chronoclone'}, 'language_info': {'name':'python'}})
nbf.write(book, BASE / '0001_gene_grounded_program_annotations.ipynb')
