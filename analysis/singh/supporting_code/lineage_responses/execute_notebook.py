"""Execute the new notebook with the existing local chronoclone IPC kernel."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

p=Path(sys.argv[1]).resolve()
book=nbformat.read(p,as_version=4)
client=NotebookClient(book,kernel_name='chronoclone',timeout=900,resources={'metadata':{'path':str(p.parent)}})
client.km=client.create_kernel_manager()
client.km.transport='ipc'
client.km.ip='/private/tmp/chronoclone-lineage-state-kernel'
try:
    client.execute()
finally:
    nbformat.write(book,p)
print(p)
