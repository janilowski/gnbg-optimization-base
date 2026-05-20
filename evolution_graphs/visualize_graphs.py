"""Visualize AST graph statistics for generated BBO candidates.

The original script mixed BP, TSP, and BBO-specific assumptions in one module.
This version is a small CLI focused on the current BBO candidate corpus. It keeps
fitness-aware plots when a fitness column exists, and otherwise falls back to
unsupervised projections and feature/complexity evolution plots.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale


DEFAULT_CSV = Path("ast/graphstats_BBO.csv")
DEFAULT_OUT_DIR = Path("ast/imgBBO")

COMPLEXITY_COLS = {
    "mean_complexity",
    "total_complexity",
    "mean_token_count",
    "total_token_count",
    "mean_parameter_count",
    "total_parameter_count",
}

BASE_METADATA_COLS = {
    "schema_version",
    "corpus",
    "path",
    "model",
    "LLM",
    "exp_dir",
    "filename",
    "candidate_id",
    "alg_id",
    "source_index",
    "sequence_kind",
    "parent_id",
    "parent_ids",
    "lineage_available",
    "fitness",
    "fitness_source",
    "has_fitness",
    "parse_ok",
    "graph_ok",
    "complexity_ok",
    "error",
    "code_diff",
    "gen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCA/t-SNE and feature-evolution plots from BBO AST graph stats."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input graphstats CSV (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for generated figures (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--problem",
        default="BBO",
        help="Problem label used in plot titles and filenames (default: BBO).",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Column used to group candidates. Defaults to model, then LLM, then exp_dir.",
    )
    parser.add_argument(
        "--sequence-col",
        default="alg_id",
        help="Column used as pseudo-evolution order (default: alg_id).",
    )
    parser.add_argument(
        "--fitness-col",
        default="fitness",
        help="Optional fitness column. Fitness-aware plots are skipped if absent or empty.",
    )
    parser.add_argument(
        "--include-complexity-in-projection",
        action="store_true",
        help="Include complexity metrics in PCA/t-SNE features.",
    )
    parser.add_argument(
        "--top-evolution-features",
        type=int,
        default=24,
        help="Maximum number of per-feature evolution plots to write (default: 24).",
    )
    return parser.parse_args()


def prettify(name: str) -> str:
    return re.sub(r"_+", " ", name).title()


def safe_name(name: object) -> str:
    value = str(name)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "group"


def choose_group_col(data: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in data.columns:
        return requested
    for column in ("model", "LLM", "exp_dir"):
        if column in data.columns:
            return column
    data["group"] = "BBO"
    return "group"


def ensure_sequence_col(data: pd.DataFrame, requested: str) -> str:
    if requested in data.columns:
        data[requested] = pd.to_numeric(data[requested], errors="coerce")
        if data[requested].isna().all():
            data[requested] = np.arange(len(data))
        else:
            data[requested] = data[requested].fillna(method="ffill").fillna(0)
        return requested

    data["sequence"] = np.arange(len(data))
    return "sequence"


def has_usable_fitness(data: pd.DataFrame, fitness_col: str) -> bool:
    if fitness_col not in data.columns:
        return False
    data[fitness_col] = pd.to_numeric(data[fitness_col], errors="coerce")
    return data[fitness_col].notna().any()


def metadata_columns(data: pd.DataFrame, group_col: str, sequence_col: str, fitness_col: str) -> set[str]:
    metadata = set(BASE_METADATA_COLS)
    metadata.update({group_col, sequence_col, fitness_col})
    metadata.update(column for column in data.columns if column.startswith("note_"))
    metadata.update(column for column in data.columns if data[column].dtype == "object")
    metadata.update(column for column in data.columns if data[column].dtype == "bool")
    return metadata


def numeric_feature_frame(
    data: pd.DataFrame,
    metadata: set[str],
    include_complexity: bool,
) -> pd.DataFrame:
    excluded = metadata if include_complexity else metadata | COMPLEXITY_COLS
    candidate_features = data.drop(columns=[c for c in excluded if c in data.columns])
    features = candidate_features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("No numeric feature columns are available for visualization.")

    features = features.fillna(features.median(numeric_only=True)).fillna(0)
    varying_cols = [column for column in features.columns if features[column].nunique() > 1]
    return features[varying_cols] if varying_cols else features


def scaled_features(features: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(features)


def add_pca_projection(data: pd.DataFrame, features_scaled: np.ndarray, problem: str) -> None:
    components = min(2, features_scaled.shape[0], features_scaled.shape[1])
    if components < 1:
        data["pca_x"] = 0.0
        data["pca_y"] = 0.0
        return

    projection = PCA(n_components=components).fit_transform(features_scaled)
    data["pca_x"] = projection[:, 0]
    data["pca_y"] = projection[:, 1] if components > 1 else 0.0
    print(f"{problem} PCA components: {components}")


def add_tsne_projection(data: pd.DataFrame, features_scaled: np.ndarray) -> bool:
    n_samples = features_scaled.shape[0]
    if n_samples < 3:
        data["tsne_x"] = np.nan
        data["tsne_y"] = np.nan
        return False

    perplexity = min(30, max(2, n_samples // 10))
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)

    projection = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features_scaled)
    data["tsne_x"], data["tsne_y"] = projection[:, 0], projection[:, 1]
    return True


def save_histogram(data: pd.DataFrame, column: str, out_path: Path, title: str) -> None:
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=20, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(prettify(column))
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_projection(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    out_path: Path,
    title: str,
    fitness_col: str | None = None,
) -> None:
    plt.figure(figsize=(10, 8))
    kwargs = {
        "x": x_col,
        "y": y_col,
        "hue": group_col,
        "data": data,
        "palette": "tab10",
        "s": 28,
    }
    if fitness_col is not None:
        kwargs["size"] = fitness_col
        kwargs["sizes"] = (20, 150)
    ax = sns.scatterplot(**kwargs)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title=prettify(group_col))
    plt.title(title)
    plt.xlabel(prettify(x_col))
    plt.ylabel(prettify(y_col))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_group_fitness_projection(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    fitness_col: str,
    out_dir: Path,
    prefix: str,
) -> None:
    for group_value in sorted(data[group_col].dropna().unique()):
        subset = data[data[group_col] == group_value]
        plt.figure(figsize=(7, 6))
        plt.scatter(subset[x_col], subset[y_col], c=subset[fitness_col], cmap="viridis", s=24)
        plt.colorbar(label=prettify(fitness_col))
        plt.title(f"{prefix} Colored By Fitness - {group_value}")
        plt.xlabel(prettify(x_col))
        plt.ylabel(prettify(y_col))
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_Fitness_{safe_name(group_value)}.png")
        plt.close()


def save_feature_evolution(
    data: pd.DataFrame,
    features: pd.DataFrame,
    group_col: str,
    sequence_col: str,
    out_dir: Path,
    limit: int,
) -> None:
    evolution_dir = out_dir / "evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)

    feature_order = features.var(numeric_only=True).sort_values(ascending=False).index[:limit]
    for feature in feature_order:
        plt.figure(figsize=(10, 6))
        for group_value in sorted(data[group_col].dropna().unique()):
            subset = data[data[group_col] == group_value].sort_values(sequence_col)
            plt.plot(subset[sequence_col], subset[feature], label=str(group_value), alpha=0.8)
        plt.title(f"Evolution of {prettify(feature)}")
        plt.xlabel(prettify(sequence_col))
        plt.ylabel(prettify(feature))
        plt.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(evolution_dir / f"Evolution_{safe_name(feature)}.png")
        plt.close()


def save_complexity_evolution(
    data: pd.DataFrame,
    group_col: str,
    sequence_col: str,
    out_dir: Path,
) -> None:
    available = [column for column in COMPLEXITY_COLS if column in data.columns]
    if not available:
        return

    complexity_dir = out_dir / "complexity"
    complexity_dir.mkdir(parents=True, exist_ok=True)
    for column in sorted(available):
        values = pd.to_numeric(data[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        data[column] = values
        plt.figure(figsize=(10, 6))
        for group_value in sorted(data[group_col].dropna().unique()):
            subset = data[data[group_col] == group_value].sort_values(sequence_col)
            plt.plot(subset[sequence_col], subset[column], label=str(group_value), alpha=0.8)
        plt.title(f"Evolution of {prettify(column)}")
        plt.xlabel(prettify(sequence_col))
        plt.ylabel(prettify(column))
        plt.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(complexity_dir / f"Evolution_{safe_name(column)}.png")
        plt.close()


def save_fitness_feature_analysis(
    data: pd.DataFrame,
    features: pd.DataFrame,
    group_col: str,
    fitness_col: str,
    out_dir: Path,
    problem: str,
) -> None:
    if data[fitness_col].notna().sum() < 5 or data[fitness_col].nunique(dropna=True) < 2:
        print("Skipping supervised feature analysis: not enough varied fitness values.")
        return

    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    target = minmax_scale(data[fitness_col].fillna(data[fitness_col].median()))

    correlations = features.corrwith(pd.Series(target, index=features.index))
    correlations = correlations.dropna().sort_values(key=np.abs, ascending=False)
    if not correlations.empty:
        plt.figure(figsize=(12, 7))
        sns.barplot(x=correlations.index[:30], y=correlations.values[:30])
        plt.xticks(rotation=90)
        plt.title("Top Feature Correlations With Fitness")
        plt.ylabel("Correlation")
        plt.tight_layout()
        plt.savefig(features_dir / f"{problem}_Feature_Fitness_Correlation.png")
        plt.close()

    if len(features) < 10:
        print("Skipping Random Forest feature analysis: fewer than 10 samples.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=42,
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{problem} Random Forest fitness model: MSE={mse:.4f} R2={r2:.4f}")

    importances = pd.Series(model.feature_importances_, index=features.columns)
    importances = importances.sort_values(ascending=False).head(30)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=importances.index, y=importances.values)
    plt.xticks(rotation=90)
    plt.title(f"Random Forest Feature Importances By {prettify(group_col)}")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(features_dir / f"{problem}_Random_Forest_Feature_Importance_{r2:.4f}.png")
    plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.csv)
    data = data.replace([np.inf, -np.inf], np.nan)
    group_col = choose_group_col(data, args.group_col)
    sequence_col = ensure_sequence_col(data, args.sequence_col)
    fitness_available = has_usable_fitness(data, args.fitness_col)

    metadata = metadata_columns(data, group_col, sequence_col, args.fitness_col)
    features = numeric_feature_frame(data, metadata, args.include_complexity_in_projection)
    features_scaled = scaled_features(features)
    for column in features.columns:
        data[column] = features[column]

    print(f"Loaded {len(data)} {args.problem} rows from {args.csv}")
    print(f"Grouping by {group_col}; using {len(features.columns)} numeric projection features.")
    if fitness_available:
        print(data[args.fitness_col].describe())
        save_histogram(
            data,
            args.fitness_col,
            args.out_dir / f"{args.problem}_Fitness_Histogram.png",
            f"{args.problem} Fitness Distribution",
        )
    else:
        print("Fitness column is absent or empty; skipping fitness-colored plots.")

    add_pca_projection(data, features_scaled, args.problem)
    tsne_available = add_tsne_projection(data, features_scaled)

    save_projection(
        data,
        "pca_x",
        "pca_y",
        group_col,
        args.out_dir / f"{args.problem}_PCA_By_{safe_name(group_col)}.png",
        f"{args.problem} PCA Projection",
        args.fitness_col if fitness_available else None,
    )
    if tsne_available:
        save_projection(
            data,
            "tsne_x",
            "tsne_y",
            group_col,
            args.out_dir / f"{args.problem}_tSNE_By_{safe_name(group_col)}.png",
            f"{args.problem} t-SNE Projection",
            args.fitness_col if fitness_available else None,
        )

    if fitness_available:
        save_group_fitness_projection(
            data,
            "pca_x",
            "pca_y",
            group_col,
            args.fitness_col,
            args.out_dir,
            "PCA",
        )
        if tsne_available:
            save_group_fitness_projection(
                data,
                "tsne_x",
                "tsne_y",
                group_col,
                args.fitness_col,
                args.out_dir,
                "tSNE",
            )
        save_fitness_feature_analysis(data, features, group_col, args.fitness_col, args.out_dir, args.problem)

    save_feature_evolution(
        data,
        features,
        group_col,
        sequence_col,
        args.out_dir,
        args.top_evolution_features,
    )
    save_complexity_evolution(data, group_col, sequence_col, args.out_dir)


if __name__ == "__main__":
    main()
