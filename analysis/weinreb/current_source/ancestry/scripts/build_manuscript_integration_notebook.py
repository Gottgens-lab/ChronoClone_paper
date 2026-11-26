#!/usr/bin/env python3
"""Build the executed provenance companion for the local manuscript revision.

This utility only assembles notebook cells. The notebook is the reviewable
entry point for document construction, rendering, and reproducible checks.
It is not a scientific reanalysis and cannot alter the live Google document.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat as nbf


REPOSITORY = Path("/path/to/ChronoClone")
WORK = REPOSITORY / "analysis_workspace/analyses/weinreb_metaclone_ancestry_worked_examples_20260910"
MANUSCRIPT_DIR = REPOSITORY / "analysis_workspace/manuscript_drafts/chronoclone_weinreb_manuscript_20260901"
ENVIRONMENT = Path("/path/to/ChronoClone/.runtime/chronoclone")
BUNDLE = Path(os.environ.get("CHRONOCLONE_DOCUMENT_DEPENDENCIES", str(REPOSITORY / ".runtime/documents")))
DOCUMENT_RUNTIME = Path(os.environ.get("CHRONOCLONE_DOCUMENT_RUNTIME", str(REPOSITORY / "document_runtime")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manuscript", type=Path, required=True)
    parser.add_argument("--result-table", type=Path, action="append", default=[])
    args = parser.parse_args()

    cells = []
    markdown = lambda source: cells.append(nbf.v4.new_markdown_cell(source))
    code = lambda source: cells.append(nbf.v4.new_code_cell(source))
    markdown("""# Local manuscript integration and provenance

This companion builds the local Weinreb manuscript from its edited Markdown,
renders a dated DOCX and PDF, and records the relationship between manuscript
claims, executed companion tables, and individual figure panels. It performs
document and provenance checks only; scientific calculations remain in the
ancestry and dynamics notebooks. The live Google document, paper-release
repository, and Singh material are outside this operation.

## Runtime and execution

The notebook reuses the existing MacBook-local **chronoclone** environment at
`/path/to/ChronoClone/.runtime/chronoclone`, kernel
`chronoclone`. Its Python and key package versions are printed in the setup
cell. No environment is installed or modified. Document authoring uses the
document dependencies configured by `CHRONOCLONE_DOCUMENT_DEPENDENCIES`
(Python, Node and LibreOffice), and `render_docx.py` in the directory
configured by `CHRONOCLONE_DOCUMENT_RUNTIME`.

Execute from `/path/to/ChronoClone`:

```bash
/path/to/ChronoClone/.runtime/chronoclone/bin/jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=chronoclone --ExecutePreprocessor.timeout=7200 analysis_workspace/analyses/weinreb_metaclone_ancestry_worked_examples_20260910/0003_manuscript_integration.ipynb
```

The document follows the retained manuscript layout. Each analytical figure
is supplied independently by an executed source notebook; no panels are
combined here. Every rendered document page must also pass individual visual
review before handoff. That review is recorded separately from the mechanical
checks so a successful renderer cannot be mistaken for visual approval.
""")
    code(f'''from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
from IPython.display import display

REPOSITORY = Path({str(REPOSITORY)!r})
WORK = Path({str(WORK)!r})
MANUSCRIPT_DIR = Path({str(MANUSCRIPT_DIR)!r})
SNAPSHOT_MANUSCRIPT = Path({str(args.snapshot_manuscript.resolve())!r})
RESULT_TABLES = [Path(path) for path in {list(map(lambda p: str(p.resolve()), args.result_table))!r}]
ENVIRONMENT = Path({str(ENVIRONMENT)!r})
BUNDLE = Path({str(BUNDLE)!r})
DOCUMENT_RUNTIME = Path({str(DOCUMENT_RUNTIME)!r})
BUNDLED_PYTHON = BUNDLE / "python/bin/python3"
BUNDLED_NODE = BUNDLE / "node/bin/node"
BUNDLED_SOFFICE = BUNDLE / "native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/MacOS/soffice"
assert (DOCUMENT_RUNTIME / "render_docx.py").is_file(), "Set CHRONOCLONE_DOCUMENT_RUNTIME to the directory containing render_docx.py"
MANUSCRIPT = MANUSCRIPT_DIR / "manuscript.md"
DOCX_BUILDER = MANUSCRIPT_DIR / "build_cell_reports_docx.py"
DOCX = MANUSCRIPT_DIR / "chronoclone_cell_reports_clone_ancestry_20260910.docx"
RENDER_DIR = MANUSCRIPT_DIR / "cell_reports_rendered_clone_ancestry_20260910"
PDF = RENDER_DIR / (DOCX.stem + ".pdf")
TABLES = WORK / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
previous_page_register_path = TABLES / "manuscript_rendered_pages_20260910.csv"
previous_page_register = pd.read_csv(previous_page_register_path) if previous_page_register_path.is_file() else None

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert Path(sys.prefix).resolve() == ENVIRONMENT.resolve()
for required in [BUNDLED_PYTHON, BUNDLED_NODE, BUNDLED_SOFFICE, MANUSCRIPT,
                 DOCX_BUILDER, SNAPSHOT_MANUSCRIPT, *RESULT_TABLES]:
    assert required.is_file(), required
assert RESULT_TABLES, "Pass executed source result tables to the notebook builder."
versions = {{"notebook_environment": "chronoclone", "environment_path": str(ENVIRONMENT),
            "kernel": "chronoclone", "python": sys.version.split()[0],
            **{{package: metadata.version(package) for package in ["nbformat", "nbconvert", "pandas"]}}}}
print(json.dumps(versions, indent=2))
version_probe = subprocess.run(
    [str(BUNDLED_PYTHON), "-c", "import sys, importlib.metadata as m; print(sys.version); print({{p:m.version(p) for p in ['python-docx','Pillow','pdf2image']}})"],
    capture_output=True, text=True, check=True)
print("Document runtime:", str(BUNDLED_PYTHON))
print(version_probe.stdout)
print("Bundled LibreOffice:", str(BUNDLED_SOFFICE))
''')
    markdown("""## Source manuscript and executed evidence

The preserved pre-edit Markdown is the comparison anchor. The ODE subsection
is retained verbatim. Scientific effects in the edited manuscript must point
to the executed source tables listed below. The document retains the original
independent ODE panels and cites the replacement ancestry figures directly.
""")
    code('''snapshot_text = SNAPSHOT_MANUSCRIPT.read_text(encoding="utf-8")
manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
section_start = "### ODE simulations quantify robustness to post-start variation in clonal trajectories"
section_end = "### ChronoClone separates developmental pace from lineage output in the Weinreb hematopoietic atlas"
snapshot_ode = snapshot_text.split(section_start, 1)[1].split(section_end, 1)[0]
current_ode = manuscript_text.split(section_start, 1)[1].split(section_end, 1)[0]
assert snapshot_ode == current_ode, "ODE text/panel references changed."
assert not re.search(r"cell_level.*\\.(?:png|svg)", manuscript_text)
for retired_phrase in ["2,936 genes", "798 genes", "1,308,530 pair-gene", "Day-2 cell-level metaclone-associated expression analysis"]:
    assert retired_phrase not in manuscript_text, retired_phrase
assert "Singh" not in manuscript_text

image_references = re.findall(r"!\\[\\]\\(([^)]+)\\)\\{width=([0-9.]+)in\\}", manuscript_text)
assert image_references
panel_rows = []
for image_ref, requested_width in image_references:
    panel_path = (MANUSCRIPT_DIR / image_ref).resolve()
    assert panel_path.is_file(), panel_path
    assert panel_path.is_relative_to(REPOSITORY / "analysis_workspace")
    panel_rows.append({"manuscript_reference": image_ref,
                       "path": str(panel_path.relative_to(REPOSITORY)),
                       "requested_width_in": float(requested_width),
                       "sha256": sha256(panel_path)})
panel_manifest = pd.DataFrame(panel_rows)
assert not panel_manifest["path"].duplicated().any()
panel_manifest.to_csv(TABLES / "manuscript_panel_manifest_20260910.csv", index=False)
display(panel_manifest)

source_rows = [{"path": str(path.relative_to(REPOSITORY)), "sha256": sha256(path),
                "rows": len(pd.read_csv(path))} for path in RESULT_TABLES]
source_manifest = pd.DataFrame(source_rows)
source_manifest.to_csv(TABLES / "manuscript_evidence_manifest_20260910.csv", index=False)
display(source_manifest)
print("ODE section and references preserved; retired cell-DE panels absent.")
''')
    markdown("""## Build the local document

The authoring-operation marker is recorded immediately before the first
document edit in this task. This cell invokes the preserved formatting
builder with an explicit dated output. It does not replace the earlier DOCX.
""")
    code('''build_command = [str(BUNDLED_PYTHON), str(DOCX_BUILDER), "--output", str(DOCX)]
print("Execution command:", " ".join(build_command))
build_result = subprocess.run(build_command, cwd=REPOSITORY, capture_output=True,
                              text=True, check=True)
print(build_result.stdout)
if build_result.stderr.strip():
    print(build_result.stderr)
assert DOCX.is_file() and DOCX.stat().st_size > 0
''')
    markdown("""## Render individual document pages

The packaged renderer resolves the bundled LibreOffice through the selected
Python runtime. It emits an individual PNG for each page and a PDF companion.
No page contact sheet or figure montage is constructed.
""")
    code('''render_command = [str(BUNDLED_PYTHON), str(DOCUMENT_RUNTIME / "render_docx.py"),
                  str(DOCX), "--output_dir", str(RENDER_DIR), "--emit_pdf"]
if RENDER_DIR.is_dir() and any(RENDER_DIR.iterdir()):
    previous_render_dir = WORK / "provenance" / ("manuscript_render_before_rebuild_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    previous_render_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(RENDER_DIR), str(previous_render_dir))
    print("Preserved previous rendered proof:", previous_render_dir)
render_environment = os.environ.copy()
render_environment["PATH"] = str(BUNDLE / "bin/override") + os.pathsep + render_environment.get("PATH", "")
render_environment["TMPDIR"] = "/private/tmp"
print("Execution command:", " ".join(render_command))
render_result = subprocess.run(render_command, cwd=REPOSITORY,
                              env=render_environment, capture_output=True,
                              text=True, check=True, timeout=3600)
print(render_result.stdout)
if render_result.stderr.strip():
    print(render_result.stderr)
page_images = sorted(RENDER_DIR.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[-1]))
assert PDF.is_file() and PDF.stat().st_size > 0
assert page_images
assert [int(path.stem.split("-")[-1]) for path in page_images] == list(range(1, len(page_images) + 1))
pdf_page_count = int(subprocess.run(
    [str(BUNDLED_PYTHON), "-c", "from pypdf import PdfReader; import sys; print(len(PdfReader(sys.argv[1]).pages))", str(PDF)],
    capture_output=True, text=True, check=True).stdout.strip())
assert len(page_images) == pdf_page_count, "Rendered page images and PDF page count disagree."
print("Rendered page count:", len(page_images))
print("DOCX:", DOCX)
print("PDF:", PDF)
''')
    markdown("""## Reproducible document checks and review register

This cell verifies embedded-panel counts, document text, page sequence, and
source hashes. Human visual inspection of each page is a distinct task; the
review template starts unreviewed and must be completed after opening every
individual page image. Mechanical checks do not assert that visual review
has occurred.
""")
    code('''from zipfile import ZipFile
import xml.etree.ElementTree as ET

with ZipFile(DOCX) as archive:
    document_xml = archive.read("word/document.xml")
document_tree = ET.fromstring(document_xml)
namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
              "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
embedded_images = document_tree.findall(".//a:blip", namespaces)
docx_text = " ".join(node.text or "" for node in document_tree.findall(".//w:t", namespaces))
assert len(embedded_images) == len(panel_manifest)
for retired_phrase in ["2,936 genes", "798 genes", "1,308,530 pair-gene", "Day-2 cell-level metaclone-associated expression analysis"]:
    assert retired_phrase not in docx_text, retired_phrase
assert "ChronoClone" in docx_text
assert "Singh" not in docx_text

qa_rows = [{"page": index + 1, "path": str(path.relative_to(REPOSITORY)),
            "sha256": sha256(path)} for index, path in enumerate(page_images)]
page_register = pd.DataFrame(qa_rows)
if previous_page_register is not None:
    page_comparison = previous_page_register[["page", "sha256"]].merge(
        page_register[["page", "sha256"]], on="page", how="outer", suffixes=("_before", "_current"))
    page_comparison["identical"] = page_comparison["sha256_before"] == page_comparison["sha256_current"]
    page_comparison.to_csv(TABLES / "manuscript_render_rebuild_comparison_20260910.csv", index=False)
    display(page_comparison)
page_register.to_csv(TABLES / "manuscript_rendered_pages_20260910.csv", index=False)
visual_review_input = WORK / "reports/manuscript_visual_review_approvals_20260910.csv"
if visual_review_input.is_file():
    visual_approvals = pd.read_csv(visual_review_input)
    visual_review_checks = page_register.merge(visual_approvals, on="page", how="outer", suffixes=("_rendered", "_reviewed"), validate="one_to_one")
    visual_review_checks["passed"] = visual_review_checks["sha256_rendered"] == visual_review_checks["sha256_reviewed"]
    assert visual_review_checks["passed"].all(), "Visual approval does not match current individual page images."
    assert visual_review_checks["reviewer"].notna().all()
    visual_review_checks.to_csv(TABLES / "manuscript_visual_review_validated_20260910.csv", index=False)
    display(visual_review_checks[["page", "reviewer", "passed"]])
else:
    print("Visual review receipts remain pending; mechanical checks are not visual approval.")
artifact_paths = [MANUSCRIPT, DOCX_BUILDER, DOCX, PDF,
                  WORK / "scripts/build_manuscript_integration_notebook.py",
                  TABLES / "manuscript_panel_manifest_20260910.csv",
                  TABLES / "manuscript_evidence_manifest_20260910.csv",
                  TABLES / "manuscript_rendered_pages_20260910.csv"]
artifact_paths += [path for path in [visual_review_input,
                  WORK / "reports/manuscript_visual_review_20260910.md",
                  TABLES / "manuscript_visual_review_validated_20260910.csv",
                  TABLES / "manuscript_render_rebuild_comparison_20260910.csv"] if path.is_file()]
artifact_manifest = pd.DataFrame([{"path": str(path.relative_to(REPOSITORY)),
                                   "bytes": path.stat().st_size,
                                   "sha256": sha256(path)} for path in artifact_paths])
artifact_manifest.to_csv(TABLES / "manuscript_integration_manifest_20260910.csv", index=False)
display(artifact_manifest)
display(pd.DataFrame([{"check": "ODE source preserved", "passed": snapshot_ode == current_ode},
                      {"check": "all panels embedded", "passed": len(embedded_images) == len(panel_manifest)},
                      {"check": "individual pages rendered", "passed": len(page_images) > 0},
                      {"check": "no retired cell-DE claims", "passed": "2,936 genes" not in docx_text},
                      {"check": "no Singh material", "passed": "Singh" not in docx_text}]))
print("Mechanical checks passed. Individual page visual review is recorded at handoff.")
''')

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python (chronoclone)", "language": "python", "name": "chronoclone"}
    notebook.metadata.language_info = {"name": "python"}
    nbf.validate(notebook)
    output = WORK / "0003_manuscript_integration.ipynb"
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
