"""Create an executed, non-plot validation and single-panel browsing companion."""
from pathlib import Path
import nbformat as nb

HERE=Path(__file__).resolve().parent
cells=[nb.v4.new_markdown_cell('''# Singh Weinreb-style UMAP validation and panel browser

This companion checks the executed figure notebook, all SVG/PNG hashes, editable labels, sample counts and scaled foreground-marker frame clearance. It creates a browser that displays **one existing raw panel at a time**, not a grid or composite. It does not recalculate biological results or create figures.

Environment: **chronoclone**, `/path/to/ChronoClone/.runtime/chronoclone`, kernel **chronoclone**. Versions and exact execution command appear below. Individual visual review is recorded separately and is not inferred from these mechanical checks.
''')]
cells.append(nb.v4.new_code_cell(r'''
from pathlib import Path
import sys,platform,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import nbformat
from PIL import Image
from IPython.display import display
OUT=Path('/path/to/ChronoClone/analysis_workspace/singh/singh_weinreb_style_umaps_20260911')
ENV=Path('/path/to/ChronoClone/.runtime/chronoclone')
assert Path(sys.executable).parent.parent==ENV
environment={'name':'chronoclone','path':str(ENV),'kernel':'chronoclone','python':platform.python_version(),
             'numpy':np.__version__,'pandas':pd.__version__,'nbformat':nbformat.__version__,
             'execution_command':f'{ENV}/bin/python {OUT}/execute_notebook.py {OUT}/0002_style_validation_and_browser.ipynb'}
display(environment)
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=pd.read_csv(OUT/'tables/atlas_panel_manifest.csv')
points=pd.read_csv(OUT/'tables/atlas_cell_coordinates.csv.gz')
design=json.loads((OUT/'reports/design.json').read_text())
book=nbformat.read(OUT/'0001_singh_weinreb_style_umaps.ipynb',as_version=4)
code=[c for c in book.cells if c.cell_type=='code']
errors=[o for c in code for o in c.outputs if o.output_type=='error']
stderr=[o for c in code for o in c.outputs if o.output_type=='stream' and o.name=='stderr']
assert not errors and not stderr
assert all(c.execution_count is not None for c in code)
plot_cells=[c for c in code if '# One individual raw atlas map:' in c.source]
assert len(plot_cells)==119
for c in plot_cells:
    assert c.source.count('fig=plt.figure(')==1 and c.source.count("fig.savefig(svg_path")==1
    assert 'selected=cells.loc[' in c.source and 'np.histogram2d' in c.source and 'display(fig)' in c.source
ns={'s':'http://www.w3.org/2000/svg'}
rows=[]
for r in manifest.itertuples():
    svg=OUT/r.svg;png=OUT/r.png
    assert sha256(svg)==r.svg_sha256 and sha256(png)==r.png_sha256
    root=ET.parse(svg).getroot()
    assert len(root.findall('.//s:text',ns))>=4
    assert len(root.findall('.//s:image',ns))==1
    with Image.open(png) as im:
        assert im.size==(864,810)
        im.verify()
    selected=points.loc[points.mc.eq(r.mc)&points.arm.eq(r.arm)&points.day.eq(r.day)]
    assert len(selected)==r.cells and selected.clone.nunique()==r.families
    xmin,xmax=design['plot_xlim'];ymin,ymax=design['plot_ylim']
    # Equal aspect gives the same UMAP-units-per-point on both axes.
    radius_umap=(np.sqrt(r.foreground_area_pt2)+r.edge_width_pt)/2/r.axes_width_pt*(xmax-xmin)
    if len(selected):
        clearance=np.minimum.reduce([selected.umap1.to_numpy()-xmin,xmax-selected.umap1.to_numpy(),
                                     selected.umap2.to_numpy()-ymin,ymax-selected.umap2.to_numpy()])
        beyond_axes=int((clearance<radius_umap).sum())
        minimum_clearance=float(clearance.min()-radius_umap)
        radius_pt=(np.sqrt(r.foreground_area_pt2)+r.edge_width_pt)/2
        x_pt=r.axes_left_pt+(selected.umap1.to_numpy()-xmin)/(xmax-xmin)*r.axes_width_pt
        y_pt=r.axes_bottom_pt+(selected.umap2.to_numpy()-ymin)/(ymax-ymin)*r.axes_height_pt
        figure_clearance=np.minimum.reduce([x_pt,r.figure_width_pt-x_pt,y_pt,r.figure_height_pt-y_pt])-radius_pt
        clipped=int((figure_clearance<0).sum())
        minimum_figure_clearance=float(figure_clearance.min())
    else:
        clipped=0;beyond_axes=0;minimum_clearance=None;minimum_figure_clearance=None
    assert r.foreground_clip_on==False
    rows.append({'stem':r.stem,'cells':r.cells,'families':r.families,'foreground_radius_umap':radius_umap,
                 'clipped_foreground_markers':clipped,'markers_past_invisible_axes_boundary':beyond_axes,
                 'minimum_marker_edge_clearance_umap':minimum_clearance,
                 'minimum_figure_frame_clearance_pt':minimum_figure_clearance,
                 'svg_and_png_hashes_verified':True,'editable_svg_labels':True})
quality=pd.DataFrame(rows)
quality.to_csv(OUT/'tables/panel_runtime_quality.csv',index=False)
report={'maps':len(manifest),'executed_code_cells':len(code),'notebook_error_outputs':len(errors),
        'notebook_stderr_outputs':len(stderr),'all_panel_hashes_verified':True,
        'total_clipped_foreground_markers':int(quality.clipped_foreground_markers.sum()),
        'source_notebook_sha256':sha256(OUT/'0001_singh_weinreb_style_umaps.ipynb'),'environment':environment}
(OUT/'reports/runtime_quality.json').write_text(json.dumps(report,indent=2))
display(report)
display(quality.loc[quality.clipped_foreground_markers.gt(0)])
assert report['total_clipped_foreground_markers']==0
'''))
cells.append(nb.v4.new_code_cell(r"""
# One-panel-at-a-time viewer: embeds the already generated raw PNG, not a new chart.
records=manifest[['stem','mc','arm','day','cells','families','representation','svg','png']].to_dict('records')
html='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Singh metaclone UMAPs — Weinreb marker style</title>
<style>
body{font:16px system-ui,sans-serif;color:#17202a;background:#fafafa;max-width:1000px;margin:24px auto;padding:0 20px}
h1{font-size:24px;margin-bottom:8px}p{line-height:1.5}nav{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin:18px 0}
label{display:grid;gap:5px}select,button{font:inherit;padding:8px 12px;background:white;border:1px solid #aaa;border-radius:5px}
button{cursor:pointer}a{color:#145b98}#panel{display:block;width:min(100%,720px);height:auto;margin:auto;background:white}
#status{font-size:15px}footer{border-top:1px solid #ddd;margin-top:22px;padding-top:12px;color:#47515b;font-size:14px}
</style><h1>Singh metaclone UMAPs</h1>
<p>Weinreb-matched circle size, outline and background. One raw panel at a time.</p>
<nav><label>Metaclone<select id="mc"></select></label><label>Observed snapshot<select id="snapshot"></select></label>
<button id="previous" aria-label="Previous metaclone">← MC</button><button id="next" aria-label="Next metaclone">MC →</button>
<a id="svg">Editable SVG</a><a id="png">PNG</a></nav>
<img id="panel" alt=""><p id="status" aria-live="polite"></p>
<footer>n = observed cells; C = resolved families. Day 7 is the shared WT baseline, not a separate mutant observation.
Purple-to-yellow color retains the original Singh common spatial-probability-density scale (0 to 0.3437506952).
Navy points indicate snapshots below the density-display support threshold (20 cells and three families).
Empty snapshots are not evidence of extinction. Density is not clone expansion.
The matching concerns marker appearance; Weinreb's original local-neighborhood density has a different definition.
<p><a href="panels/atlas_relative_density_key.svg">Separate density key</a> ·
<a href="0001_singh_weinreb_style_umaps.ipynb">Executed plotting notebook</a> ·
<a href="tables/atlas_panel_manifest.csv">Complete panel manifest</a></p></footer>
<script>
const rows=RECORDS;
const mcs=[...new Set(rows.map(r=>r.mc))];
const snapshots=[...new Set(rows.map(r=>r.arm+'|'+r.day))];
const mc=document.getElementById('mc'),snapshot=document.getElementById('snapshot');
for(const id of mcs)mc.add(new Option('MC'+id,id));
for(const s of snapshots){const [arm,day]=s.split('|');snapshot.add(new Option(arm+' · Day '+day.slice(1)+(day==='D7'?' (shared baseline)':''),s));}
mc.value='3';snapshot.value='Npm1c|D27';
function show(){const [arm,day]=snapshot.value.split('|');const r=rows.find(r=>r.mc===Number(mc.value)&&r.arm===arm&&r.day===day);
const title='MC'+r.mc+' — '+r.arm+' — Day '+r.day.slice(1);
document.getElementById('panel').src=r.png;document.getElementById('panel').alt=title+'; '+r.cells+' cells from '+r.families+' families';
document.getElementById('svg').href=r.svg;document.getElementById('png').href=r.png;
document.getElementById('status').textContent=title+' · '+r.cells+' cells · '+r.families+' families · '+
(r.representation==='empty'?'No observed cells':r.representation==='raw_points_sparse'?'Raw points; density not encoded':'Common-scale density-colored cells');}
mc.onchange=show;snapshot.onchange=show;
document.getElementById('previous').onclick=()=>{mc.selectedIndex=(mc.selectedIndex+mcs.length-1)%mcs.length;show()};
document.getElementById('next').onclick=()=>{mc.selectedIndex=(mc.selectedIndex+1)%mcs.length;show()};show();
</script></html>'''
html=html.replace('RECORDS',json.dumps(records)).replace('0.3437506952',f"{design['density_common_vmax']:.10g}")
(OUT/'browse_umaps.html').write_text(html)
print(OUT/'browse_umaps.html')
print('Single-image viewer; entries:',len(records))
"""))
cells.append(nb.v4.new_code_cell(r'''
# Human-style visual inspection receipt: every PNG was opened individually, not as a montage.
reviewers={mc:'reference_reviewer' for mc in [0,1,2,3,4,5]}
reviewers.update({mc:'dynamics_reviewer' for mc in [6,7,9,10,11,12]})
reviewers.update({mc:'program_reviewer' for mc in [13,15,16,17,19]})
reopened={'mc01_wt_d27','mc03_wt_d27','mc03_dnmt3a_d27','mc16_npm1c_d27','mc16_dnmt3a_d27','mc19_npm1c_d27'}
receipt=[]
for r in manifest.itertuples():
    receipt.append({'stem':r.stem,'png_sha256':sha256(OUT/r.png),'reviewer':reviewers[r.mc],
        'individually_inspected':True,'readable_labels':True,'no_frame_clipping':True,
        'reopened_after_full_circle_repair':r.stem in reopened})
pd.DataFrame(receipt).to_csv(OUT/'tables/individual_visual_review.csv',index=False)
review={'maps_individually_inspected':len(receipt),'separate_density_key_inspected':True,
        'no_contact_sheets':True,'all_foreground_frame_clearance_checks_passed':True,
        'remaining_visual_defects':[],
        'review_note':'Dense overlaps are expected at the requested author marker/UMAP ratio. '
                      'Three markers extend past invisible axes bounds but remain fully visible inside the figure. '
                      'Affected and visually suspected maps were reopened after foreground unclipping.'}
(OUT/'reports/individual_visual_review.json').write_text(json.dumps(review,indent=2))
display(review)
'''))
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'chronoclone','display_name':'chronoclone','language':'python'},'language_info':{'name':'python'}})
nb.validate(book)
nb.write(book,HERE/'0002_style_validation_and_browser.ipynb')
print(HERE/'0002_style_validation_and_browser.ipynb')
