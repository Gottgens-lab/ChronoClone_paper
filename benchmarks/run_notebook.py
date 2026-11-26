"""Execute one benchmark notebook, retaining its visible outputs."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=Path)
parser.add_argument("--kernel", default="chronoclone")
args = parser.parse_args()
path = args.notebook.resolve()
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=1800, kernel_name=args.kernel,
                        resources={"metadata": {"path": str(path.parent)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
print(f"Executed {path}: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells")
