"""Download checksum-registered manuscript assets; authentication stays with gh."""
from pathlib import Path
import argparse, hashlib, json, subprocess, tempfile, urllib.request, shutil

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',action='append',help='Asset id; may be repeated. Defaults to all assets.')
    parser.add_argument('--public',action='store_true',help='Use public HTTPS instead of authenticated gh.')
    args=parser.parse_args()
    registry=json.loads((ROOT/'data/assets.json').read_text())
    if args.public and registry.get('draft'):
        raise SystemExit('This snapshot uses a private draft release. Use authenticated gh without --public.')
    ids=set(args.asset or [a['id'] for a in registry['assets']])
    assert ids <= {a['id'] for a in registry['assets']}, 'Unknown asset id'
    for asset in registry['assets']:
        if asset['id'] not in ids:continue
        relative=Path(asset['destination'])
        assert not relative.is_absolute() and '..' not in relative.parts, 'Unsafe destination'
        dest=ROOT/asset['destination']
        if dest.exists():
            if digest(dest)!=asset['sha256']:raise SystemExit(f'Existing input has a different checksum: {dest}')
            print('Already verified:',dest);continue
        assert dest.parent.resolve().is_relative_to(ROOT.resolve()), 'Unsafe destination parent'
        with tempfile.TemporaryDirectory(prefix='chronoclone-download-') as temp:
            target=Path(temp)/asset['name']
            if args.public:
                with urllib.request.urlopen(asset['url'],timeout=120) as response,target.open('wb') as out:shutil.copyfileobj(response,out)
            else:
                subprocess.run(['gh','release','download',registry['tag'],'--repo',registry['repository'],'--pattern',asset['name'],'--dir',temp],check=True)
            if target.stat().st_size!=asset['bytes'] or digest(target)!=asset['sha256']:raise SystemExit(f'Asset checksum mismatch: {asset["name"]}')
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(target),dest)
        print('Downloaded and verified:',dest)

if __name__=='__main__':main()
