"""Check the repository tree without installing the scientific runtime."""
from pathlib import Path
import csv, hashlib, json, re, sys, os

ROOT=Path(__file__).resolve().parents[1]
EXEMPT={'MANIFEST.sha256.tsv','provenance/validation_summary.json','provenance/remote_sync.json'}
def shipped_files():
    assets={a['destination'] for a in json.loads((ROOT/'data/assets.json').read_text())['assets']}
    for base,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','.ipynb_checkpoints','generated','.reproduction','external'}]
        for name in files:
            p=Path(base)/name;rel=p.relative_to(ROOT).as_posix()
            if name in {'.git','.DS_Store'} or name.endswith(('.pyc','.log')) or rel in assets or rel in EXEMPT:continue
            yield p
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify():
    from check_publication_tree import verify as verify_labels
    label_report=verify_labels()
    checks=[]
    def require(condition,description):
        checks.append({'check':description,'passed':bool(condition)})
    entries=list(csv.DictReader((ROOT/'MANIFEST.sha256.tsv').open(),delimiter='\t'))
    expected={r['path'] for r in entries}
    actual={p.relative_to(ROOT).as_posix() for p in shipped_files()}
    require(expected==actual,f'Exact inventory; unlisted={sorted(actual-expected)}, missing={sorted(expected-actual)}')
    for item in entries:
        p=ROOT/item['path']
        require(p.is_file(),f'Exists: {item["path"]}')
        if p.is_file():require(p.stat().st_size==int(item['bytes']) and sha(p)==item['sha256'],f'Checksum: {item["path"]}')
        require(int(item['bytes'])<100*1024*1024,f'Ordinary Git size: {item["path"]}')
    require(not (ROOT/'src/chronoclone').exists(),'No installable ChronoClone package tree')
    require(not (ROOT/'pyproject.toml').exists(),'No second package distribution')
    require(all((ROOT/p).exists() for p in ['analysis/weinreb/README.md','analysis/singh/README.md','benchmarks/ode_noise/README.md']),'Study navigation exists')
    figures=list(csv.DictReader((ROOT/'figures/figure_index.csv').open()))
    require(len(figures)>=40,'Current figure panels indexed (Figures 1-4, S1-S4)')
    require({r['figure'] for r in figures}=={f'Figure {f}' for f in ['1','2','3','4','S1','S2','S3','S4']},'Every current figure represented')
    for r in figures:
        for col,key in (('svg','sha256_svg'),('png','sha256_png')):
            p=ROOT/r[col];require(p.is_file() and sha(p)==r[key],f'Panel {r["panel"]}: {r[col]}')
        require((ROOT/r['producing_notebook']).is_file(),f'Producing notebook for {r["panel"]}: {r["producing_notebook"]}')
    shipped={f.name for f in (ROOT/'figures').rglob('*') if f.suffix in {'.svg','.png'}}
    indexed={Path(r[c]).name for r in figures for c in ('svg','png')}
    require(shipped==indexed,f'Every file in figures/ is indexed; extra={sorted(shipped-indexed)}')
    require(not (ROOT/'manuscript').exists(),'No manuscript folder')
    text_like=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if '.git' not in p.parts and p.is_file() and p.name.lower().startswith('chronoclone_v') and p.suffix.lower() in {'.docx','.pdf','.md','.html','.txt','.doc'}]
    require(not text_like,f'No manuscript text in the repository: {text_like}')
    figtab=json.loads((ROOT/'provenance/figures_and_tables.json').read_text())
    for r in figtab['tables']:
        p=ROOT/r['file'];require(p.is_file() and sha(p)==r['sha256'],f'Supplementary table: {r["file"]}')
    require(all((ROOT/'supplementary_tables'/f'Table_S{i}.xlsx').is_file() for i in range(1,7)) and (ROOT/'supplementary_tables/Key_Resources.xlsx').is_file(),'Supplementary workbooks S1-S6 and Key Resources')
    for record in ['analysis/weinreb/provenance/figure_panels.json','analysis/singh/provenance/figure_panels.json']:
        for r in json.loads((ROOT/record).read_text())['files']:
            p=ROOT/r['destination'];require(p.is_file() and sha(p)==r['sha256'],f'Figure-source provenance: {r["destination"]}')
    notebook_rows=[]
    for row in entries:
        if not row['path'].endswith('.ipynb'):continue
        nb=json.loads((ROOT/row['path']).read_text())
        cells=[c for c in nb['cells'] if c['cell_type']=='code']
        errors=sum(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[]))
        historical=any(v in row['path'] for v in ['/source_notebooks/','/current_source/','/supporting_inputs/','/upstream/','/source/'])
        notebook_rows.append({'path':row['path'],'code_cells':len(cells),'executed_cells':sum(c.get('execution_count') is not None for c in cells),'saved_errors':errors,'historical_source':historical})
        require(nb.get('nbformat')==4,f'Notebook format: {row["path"]}')
        require(errors==0,f'No saved errors: {row["path"]}')
        if not historical:
            require(all(c.get('execution_count') is not None for c in cells if str(c.get('source','')).strip()),f'Executed entrypoint: {row["path"]}')
    atlas=list((ROOT/'analysis/singh/atlas/svg').glob('*.svg'))
    require(len(atlas)>=119,'Individual Singh atlas sources present')
    assets=json.loads((ROOT/'data/assets.json').read_text())
    require(len(assets['assets'])>=2,'Large benchmark and Weinreb inputs registered')
    failed=[c for c in checks if not c['passed']]
    result={'checks':len(checks),'passed':len(checks)-len(failed),'failed':failed,'files':len(entries),'bytes':sum(int(r['bytes']) for r in entries),'figure_panels':len(figures),'supplementary_tables':len(figtab['tables']),'publication_labels':label_report,'notebooks':notebook_rows}
    print(json.dumps({k:v for k,v in result.items() if k!='notebooks'},indent=2))
    if failed:raise AssertionError(f'{len(failed)} integrity checks failed')
    return result

if __name__=='__main__':verify()
