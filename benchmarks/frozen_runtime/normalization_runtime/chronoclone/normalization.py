"""
normalization.py
----------------
Normalize clone count tables to proportions per sample per time point.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def normalize_clone_table(
    final_table: pd.DataFrame,
    times_order: Optional[List[str]] = None,
    features_order: Optional[List[str]] = None,
    time_prefix: str = 'Day',
    feature_prefix: str = 'Cluster',
    sample_id_extractor=None,
) -> pd.DataFrame:
    """
    Normalize clone counts to proportions per sample per time point.

    For each clone (row) and time point, divides cell counts by the
    total count at that time point within the same sample group.
    Days are auto-detected from column names if times_order is not provided.
    Columns are reordered by times_order x features_order if both are given.

    Parameters
    ----------
    final_table : pd.DataFrame
        Output of build_clone_table().
    times_order : list of str, optional
        Ordered time labels (e.g. ['Day0', 'Day4', 'Day9', 'Day15']).
        If None, days are auto-detected from column names.
    features_order : list of str, optional
        Ordered feature labels (e.g. cell type list).
        If provided alongside times_order, columns are reordered.
    time_prefix : str
        Prefix used for time columns (default 'Day').
    feature_prefix : str
        Prefix used for feature columns (default 'Cluster').
    sample_id_extractor : callable, optional
        Function applied to the index Series to extract sample IDs.
        Defaults to first 2 characters: lambda idx: idx.str[:2]

    Returns
    -------
    pd.DataFrame
        Normalized table with same shape as final_table,
        optionally reordered by times_order x features_order.

    Examples
    --------
    >>> # Minimal — auto-detects days, no reordering
    >>> norm = cc.normalize_clone_table(final_table)

    >>> # Full — explicit order + column reordering
    >>> norm = cc.normalize_clone_table(
    ...     final_table,
    ...     times_order=['Day0', 'Day4', 'Day9', 'Day15'],
    ...     features_order=['HSC', 'HSPC', 'CMP', 'MEP']
    ... )
    """
    normalized_table = final_table.copy()

    # Auto-detect unique day prefixes from column names if not provided
    if times_order is None:
        times_order = sorted(set(
            col.split('_')[0]
            for col in final_table.columns
            if '_' in col
        ))

    for t in times_order:
        t_cols = [c for c in final_table.columns if c.startswith(t + '_')]
        if not t_cols:
            continue

        # Extract numeric values as float matrix
        sub_matrix = final_table[t_cols].values.astype(float)
        row_sums = sub_matrix.sum(axis=1, keepdims=True)

        # Normalize row-wise (avoiding division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_sub = np.where(row_sums > 0, sub_matrix / row_sums, sub_matrix)

        normalized_table[t_cols] = normalized_sub

    # Reorder columns if both times_order and features_order are given
    if features_order is not None:
        new_col_order = []
        for t in times_order:
            for f in features_order:
                col = f'{t}_{feature_prefix}{f}'
                if col in normalized_table.columns:
                    new_col_order.append(col)
        # Append any remaining columns not captured by the ordering
        remaining = [c for c in normalized_table.columns if c not in new_col_order]
        normalized_table = normalized_table[new_col_order + remaining]

    return normalized_table

