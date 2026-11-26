"""Check every current publication path and payload against restricted labels and personal paths.

The byte patterns are encoded so the validator itself complies with the rule.
Checks include JSON strings and decompressed ZIP/GZIP members. Git administration
is excluded because it is version-control storage, not a publication payload.
"""
from pathlib import Path
import gzip, io, json, re, zipfile

ROOT=Path(__file__).resolve().parents[1]
PATTERN=re.compile(b'|'.join(bytes.fromhex(s) for s in (
    '636c61756465','636f646578','636f656478',
    '616e746967726176697479','616e7468726f706963','6f70656e6169','63686174677074','67656d696e69','636f70696c6f74','646565707365656b','62726f777365722d757365')),re.I)
# Absolute personal and HPC paths. Matching is case-sensitive, so web addresses with a lowercase users
# segment are not matched; the public conda-forge build user (runner) is allowed.
PATHS=re.compile(bytes.fromhex('2f55736572732f')+b'(?!runner/)|'+b'|'.join(bytes.fromhex(s) for s in (
    '2f7264732f757365722f','2f7264732f70726f6a6563742f')))
def restricted(data):return bool(PATTERN.search(data) or PATHS.search(data))

def contains(stream):
    tail=b''
    for block in iter(lambda:stream.read(1024*1024),b''):
        data=tail+block
        if restricted(data):return True
        tail=data[-16:]
    return False

def verify(root=ROOT):
    failures=[];files=0;directories=0;members=0
    def check(data,label):
        if isinstance(data,str):data=data.encode('utf8')
        if restricted(data):failures.append(label)
    def strings(value,label):
        if isinstance(value,str):check(value,label)
        elif isinstance(value,list):
            for i,v in enumerate(value):strings(v,f'{label}[{i}]')
        elif isinstance(value,dict):
            for k,v in value.items():check(k,label);strings(v,f'{label}.{k}')
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root).as_posix()
        if '.git' in p.relative_to(root).parts:continue
        check(rel,f'Path: {rel}')
        if p.is_symlink():check(str(p.readlink()),f'Link target: {rel}')
        if p.is_dir():directories+=1;continue
        if not p.is_file():continue
        files+=1
        with p.open('rb') as f:
            if contains(f):failures.append(f'Bytes: {rel}')
        if p.suffix in {'.json','.ipynb'}:
            strings(json.loads(p.read_text()),f'JSON: {rel}')
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as archive:
                for info in archive.infolist():
                    members+=1;check(info.filename,f'Archive path: {rel}:{info.filename}')
                    with archive.open(info) as f:
                        if contains(f):failures.append(f'Archive bytes: {rel}:{info.filename}')
        elif p.suffix=='.gz':
            members+=1
            with gzip.open(p,'rb') as f:
                if contains(f):failures.append(f'Decoded gzip: {rel}')
    result={'files_checked':files,'directories_checked':directories,'archive_members_checked':members,
            'restricted_label_matches':len(failures),'failures':failures}
    if failures:raise AssertionError(json.dumps(result,indent=2))
    return result

if __name__=='__main__':print(json.dumps(verify(),indent=2))
