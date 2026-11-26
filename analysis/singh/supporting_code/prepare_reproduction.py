"""Materialize a source-layout workspace; never execute or refit analyses.

Default: exact copies only. --adapt-paths prepares fresh, unexecuted notebook and
Python copies using only the explicitly declared original-prefix replacements.
"""
from pathlib import Path
import argparse, hashlib, json, shutil

OLD_ROOT='/path/to/ChronoClone'
OLD_ENV='/path/to/ChronoClone/.runtime/chronoclone'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--external-map',type=Path,help='JSON object mapping original workspace-relative input names to exact local files')
    parser.add_argument('--adapt-paths',action='store_true')
    parser.add_argument('--environment-prefix',type=Path,default=Path(OLD_ENV))
    parser.add_argument('--additional-map',type=Path,action='append',default=[],help='Additional source_file_map.json, with paths relative to repository root')
    parser.add_argument('--no-default-map',action='store_true',help='Use only explicit --additional-map inputs; exclude the default Singh source and external registries')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--link-external',action='store_true',help='Link verified external data files into the new workspace instead of duplicating them; keep those source files available and read-only')
    args=parser.parse_args()
    study=Path(__file__).resolve().parents[1]
    repo=study.parents[1]
    rows=[] if args.no_default_map else json.loads((study/'provenance/copy_plan.json').read_text())['files']
    files={r['source']:(study/r['destination'],r['sha256']) for r in rows}
    for manifest_path in args.additional_map:
        extra=json.loads(manifest_path.read_text())
        if isinstance(extra,dict):extra=extra['files']
        for r in extra:
            key=r['source_workspace_path'];value=(repo/r['path'],r['sha256'])
            if key in files and files[key][1]!=value[1]:raise ValueError('Conflicting source: '+key)
            files[key]=value
    external=[] if args.no_default_map else json.loads((study/'provenance/external_inputs.json').read_text())['files']
    external_keys={r['source'] for r in external}
    supplied=json.loads(args.external_map.read_text()) if args.external_map else {}
    missing=[]
    for r in external:
        key=r['source']
        if key not in supplied:missing.append(key);continue
        files[key]=(Path(supplied[key]).expanduser().resolve(),r['sha256'])
    bad=[]
    if not files and not external:raise SystemExit('Select at least one source map; nothing was written.')
    for original,(source,expected) in files.items():
        if not source.is_file() or sha(source)!=expected:bad.append(original)
    report={'missing_external_inputs':missing,'missing_or_mismatched_files':bad,'materialized':False,'source_files':len(files)}
    if missing or bad:
        print(json.dumps(report,indent=2))
        raise SystemExit('Required exact inputs are unavailable; nothing was written.')
    if args.check_only:print(json.dumps(report,indent=2));return
    target=args.destination.expanduser().resolve()
    if target.exists():raise SystemExit('Destination must be new; refusing to overwrite.')
    env=args.environment_prefix.expanduser().absolute()
    if args.adapt_paths and not (env/'bin/python').is_file():raise SystemExit('Environment prefix must contain bin/python.')
    for original,(source,expected) in files.items():
        rel=Path(original)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe source path: '+original)
    target.mkdir(parents=True)
    changes=[]
    for original,(source,expected) in files.items():
        dest=target/original;dest.parent.mkdir(parents=True,exist_ok=True)
        if args.link_external and (original in external_keys or source.suffix=='.h5ad'):dest.symlink_to(source.resolve())
        else:shutil.copy2(source,dest)
        assert sha(dest)==expected
        if args.adapt_paths and dest.suffix in {'.py','.ipynb'}:
            if dest.suffix=='.ipynb':
                import nbformat
                book=nbformat.read(dest,as_version=4)
                for cell in book.cells:
                    cell.source=cell.source.replace(OLD_ENV,'PUBLICATION_RUNTIME_PLACEHOLDER').replace(OLD_ROOT,str(target)).replace('PUBLICATION_RUNTIME_PLACEHOLDER',str(env))
                    if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
                book.metadata['publication_preparation']={'source_sha256':expected,'status':'unexecuted_path_adapted_copy','workspace':str(target),'environment_prefix':str(env)}
                nbformat.write(book,dest)
            else:
                dest.write_text(dest.read_text().replace(OLD_ENV,'PUBLICATION_RUNTIME_PLACEHOLDER').replace(OLD_ROOT,str(target)).replace('PUBLICATION_RUNTIME_PLACEHOLDER',str(env)))
            changes.append({'path':original,'source_sha256':expected,'prepared_sha256':sha(dest)})
    report.update(materialized=True,destination=str(target),path_adaptations=changes,external_storage='symlinks' if args.link_external else 'copies',executed=False)
    (target/'REPRODUCTION_PREPARATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
