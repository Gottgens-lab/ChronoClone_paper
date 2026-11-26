from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
p=Path(sys.argv[1]).resolve()
b=nbformat.read(p,as_version=4)
try:
    NotebookClient(b,kernel_name='chronoclone',timeout=7200,resources={'metadata':{'path':str(p.parent)}}).execute()
finally:
    nbformat.write(b,p)
print(p)
