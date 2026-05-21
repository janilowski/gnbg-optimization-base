"""Find exact and near duplicates in AST projection features.

This is intentionally small and independent from the plotting code. It reads the
graphstats CSV, compares candidates by AST projection features, and writes two
CSV reports: one row per duplicate group and one row per duplicate candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path("evolution_graphs/results/graphstats_throwaways.csv")
DEFAULT_OUT_DIR = Path("evolution_graphs/results/duplicates")

AST_PROJECTION_FEATURES = [
    "Assortativity",
    "Average Eccentricity",
    "Average Shortest Path",
    "Clustering Variance",
    "Degree Entropy",
    "Degree Variance",
    "Depth Entropy",
    "Diameter",
    "Edge Density",
    "Edges",
    "Max Clustering",
    "Max Degree",
    "Max Depth",
    "Mean Clustering",
    "Mean Degree",
    "Mean Depth",
    "Min Clustering",
    "Min Degree",
    "Min Depth",
    "Nodes",
    "Radius",
    "Transitivity",
]

ID_COLUMNS = ["path", "model", "filename", "candidate_id", "alg_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify exact or almost-exact duplicates by AST projection features."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--decimals",
        type=int,
        default=8,
        help="Rounding precision for near-duplicate grouping (default: 8).",
    )
    parser.add_argument(
        "--drop-constant-features",
        action="store_true",
        help="Drop constant columns before grouping, mirroring visualization projections.",
    )
    return parser.parse_args()


def feature_frame(data: pd.DataFrame, drop_constant_features: bool) -> pd.DataFrame:
    present = [column for column in AST_PROJECTION_FEATURES if column in data.columns]
    if not present:
        raise ValueError("No AST projection feature columns were found in the CSV.")

    features = data[present].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0)
    if drop_constant_features:
        varying = [column for column in features.columns if features[column].nunique() > 1]
        features = features[varying] if varying else features
    return features


def build_groups(data: pd.DataFrame, features: pd.DataFrame, decimals: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rounded = features.round(decimals)
    keys = rounded.astype(str).agg("|".join, axis=1)

    id_columns = [column for column in ID_COLUMNS if column in data.columns]
    work = data[id_columns].copy()
    for column in ID_COLUMNS:
        if column not in work.columns:
            work[column] = ""
    work["duplicate_key"] = keys
    work["group_size"] = work.groupby("duplicate_key")["duplicate_key"].transform("size")
    duplicates = work[work["group_size"] > 1].copy()
    duplicates = duplicates.sort_values(["group_size", "duplicate_key"], ascending=[False, True])

    groups = (
        duplicates.groupby("duplicate_key", as_index=False)
        .agg(
            group_size=("duplicate_key", "size"),
            models=("model", lambda s: ",".join(sorted(set(map(str, s))))),
            candidates=("candidate_id", lambda s: ",".join(map(str, s))),
            paths=("path", lambda s: ",".join(map(str, s))),
        )
        .sort_values(["group_size", "duplicate_key"], ascending=[False, True])
    )
    return groups, duplicates


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.csv)
    features = feature_frame(data, args.drop_constant_features)
    exact_groups, exact_rows = build_groups(data, features, decimals=14)
    near_groups, near_rows = build_groups(data, features, decimals=args.decimals)
    exact_unique_rows = features.round(14).drop_duplicates().shape[0]
    near_unique_rows = features.round(args.decimals).drop_duplicates().shape[0]

    exact_groups.to_csv(args.out_dir / "exact_duplicate_groups.csv", index=False)
    exact_rows.to_csv(args.out_dir / "exact_duplicate_rows.csv", index=False)
    near_groups.to_csv(args.out_dir / "near_duplicate_groups.csv", index=False)
    near_rows.to_csv(args.out_dir / "near_duplicate_rows.csv", index=False)

    print(f"Rows: {len(data)}")
    print(f"AST features used: {len(features.columns)}")
    print(f"Unique exact feature rows: {exact_unique_rows}")
    print(f"Unique near feature rows at {args.decimals} decimals: {near_unique_rows}")
    print(f"Exact duplicate groups: {len(exact_groups)}")
    print(f"Near duplicate groups at {args.decimals} decimals: {len(near_groups)}")
    print(f"Wrote reports to {args.out_dir}")


if __name__ == "__main__":
    main()
