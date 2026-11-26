"""Run the saved-geometry display notebook in an isolated local IPC kernel."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path=Path(__file__).resolve().parent/'0001_saved_clone_umaps.ipynb'
book=nbformat.read(path,as_version=4)
client=NotebookClient(book,kernel_name='chronoclone',timeout=300,
                      resources={'metadata':{'path':str(path.parent)}})
client.km=client.create_kernel_manager()
client.km.transport='ipc'
client.km.ip='/private/tmp/chronoclone-clone-comparison-20260911-kernel'
try:
    client.execute()
finally:
    nbformat.write(book,path)
errors=[(i,o) for i,c in enumerate(book.cells) for o in c.get('outputs',[])
        if o.output_type=='error' or (o.output_type=='stream' and o.name=='stderr')]
assert not errors,errors
print(path)
