"""Execute in the existing chronoclone kernel with local IPC; preserve visible outputs."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
book = nbformat.read(path, as_version=4)
client = NotebookClient(book, kernel_name='chronoclone', timeout=600,
                        resources={'metadata': {'path': str(path.parent)}})
client.km = client.create_kernel_manager()
client.km.transport = 'ipc'
client.km.ip = '/private/tmp/chronoclone-singh-weinreb-style-kernel'
try:
    client.execute()
finally:
    nbformat.write(book, path)
print(path)
