# Data availability and provenance

## Weinreb

Original mouse LARRY data: [GEO GSE140802](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802). The processed landscape and annotations were obtained from [CLADES Figshare version 2](https://doi.org/10.6084/m9.figshare.27908142.v2), whose source record identifies CC BY 4.0. Its `Weinreb.zip` file is 846,198,775 bytes, MD5 `cbfa6a1442325300bc4ef257b2fb7d46`.

The downstream Weinreb input is the registered processed `klein_metaclone_palantir_trajectories.h5ad` with 9,714 cells. It preserves the full Day-2 HSPC expression cohort, including unassigned cells needed for label-blind program discovery. A smaller historical 775-cell expression derivative is insufficient for the current cNMF workflow and is not substituted. The exact processed checkpoint and checksum are listed in `data/assets.json`.

The public archive's named landscape and the locally registered upstream landscape have not been established as byte- or value-equivalent. Source documentation must preserve this gap; downstream checkpoint reproducibility does not resolve it.

## Singh

Original data: [GEO GSE266232](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE266232). Original analysis objects and scripts: [Zenodo concept record 14261507](https://doi.org/10.5281/zenodo.14261507), resolving to [version 14261508](https://doi.org/10.5281/zenodo.14261508), licensed CC BY 4.0 according to its metadata.

Large original sequencing and source-object archives are referenced at that deposit. This repository ships paper-specific derived tables, the resolved-family registry, preserved coordinates, fitted program checkpoints, and compact derived inputs. These are transformations of the attributed upstream study, not new sequencing data or a replacement for its complete deposit.

The paper uses 14,896 resolved cells from 1,516 families and 2,713 Day-7 HSPCs from 1,514 resolved families. Source HSC and MPP annotations are pooled for biological interpretation. Reference labels are the original 17-metaclone registry, not a later alternative partition.

## Assets and licenses

Use `data/assets.json` for exact names, sizes, SHA-256 values, and source destinations. Both large files are attached to this repository's GitHub release. Download them with `python scripts/fetch_assets.py --public` (HTTPS) or `python scripts/fetch_assets.py` (GitHub CLI), or supply local copies of the original inputs through the study preparers. Original deposited data remain governed by their upstream licenses and attribution requirements. Code licensing does not relicense third-party data, gene-set resources, or figures.
