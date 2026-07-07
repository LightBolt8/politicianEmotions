"""Cross-candidate AU heatmap and violin plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANGER_AUS = ("AU04", "AU05", "AU07", "AU23")
ANGER_LABELS = {
    "AU04": "Brow lowerer",
    "AU05": "Upper lid raiser",
    "AU07": "Lid tightener",
    "AU23": "Lip tightener",
}
# High cross-candidate variance beyond anger AUs
EXTRA_AUS = ("AU12", "AU14", "AU17", "AU26")
EXTRA_LABELS = {
    "AU12": "Lip corner puller",
    "AU14": "Dimpler",
    "AU17": "Chin raiser",
    "AU26": "Jaw drop",
}

CANDIDATES: list[tuple[str, Path]] = [
    ("Trump 2016", Path("analysis/2016/Trump/openface/Trump_clean_2016.csv")),
    ("Clinton 2016", Path("analysis/2016/Clinton/openface/Clinton_clean_2016.csv")),
    ("Trump 2024", Path("analysis/2024/Trump/openface/Trump_clean_2024_noads.csv")),
    ("Harris 2024", Path("analysis/2024/Harris/openface/Harris_clean_2024_noads.csv")),
]


def load_au_frame_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    if "success" in df.columns:
        df = df[df["success"] == 1]
    au_cols = [col for col in df.columns if re.fullmatch(r"AU\d{2}_r", col)]
    if not au_cols:
        raise ValueError(f"No AU columns in {csv_path}")
    return df[au_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def build_mean_matrix(frame_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for label, df in frame_data.items():
        rows[label] = df.mean()
    matrix = pd.DataFrame(rows).T
    matrix.columns = [col.replace("_r", "") for col in matrix.columns]
    au_order = sorted(matrix.columns, key=lambda x: int(x[2:]))
    return matrix[au_order]


def plot_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(1.5, matrix.values.max()),
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Mean intensity (0–5)"},
    )
    ax.set_title("Mean AU intensity by candidate")
    ax.set_xlabel("Action Unit")
    ax.set_ylabel("Candidate")
    ax.tick_params(axis="x", rotation=45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_violin_long_frame(
    frame_data: dict[str, pd.DataFrame],
    aus: tuple[str, ...],
    *,
    sample_per_candidate: int = 4000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for label, df in frame_data.items():
        cols = [f"{au}_r" for au in aus if f"{au}_r" in df.columns]
        subset = df[cols]
        if len(subset) > sample_per_candidate:
            idx = rng.choice(len(subset), size=sample_per_candidate, replace=False)
            subset = subset.iloc[idx]
        for au in aus:
            col = f"{au}_r"
            if col not in subset.columns:
                continue
            for value in subset[col].values:
                records.append({"candidate": label, "au": au, "intensity": float(value)})
    return pd.DataFrame.from_records(records)


def au_panel_label(au: str) -> str:
    if au in ANGER_LABELS:
        return f"{au}\n{ANGER_LABELS[au]}"
    if au in EXTRA_LABELS:
        return f"{au}\n{EXTRA_LABELS[au]}"
    return au


def plot_violins(long_df: pd.DataFrame, aus: tuple[str, ...], output_path: Path) -> None:
    candidate_order = [label for label, _ in CANDIDATES]
    palette = {
        "Trump 2016": "#dc2626",
        "Clinton 2016": "#2563eb",
        "Trump 2024": "#f97316",
        "Harris 2024": "#7c3aed",
    }

    ncols = 4
    nrows = int(np.ceil(len(aus) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.6 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, au in zip(axes_flat, aus):
        panel = long_df[long_df["au"] == au]
        sns.violinplot(
            data=panel,
            x="candidate",
            y="intensity",
            order=candidate_order,
            hue="candidate",
            palette=palette,
            cut=0,
            inner="quart",
            linewidth=0.8,
            ax=ax,
            legend=False,
        )
        ax.set_title(au_panel_label(au), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, min(5.0, panel["intensity"].quantile(0.995) * 1.15 + 0.1))
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    for ax in axes_flat[len(aus) :]:
        ax.axis("off")

    fig.suptitle("AU intensity distributions by candidate", fontsize=14, y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cross-candidate AU comparisons.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    frame_data: dict[str, pd.DataFrame] = {}
    for label, path in CANDIDATES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing OpenFace CSV: {path}")
        frame_data[label] = load_au_frame_data(path)

    mean_matrix = build_mean_matrix(frame_data)
    plot_heatmap(mean_matrix, output_dir / "au_heatmap.png")
    mean_matrix.to_csv(output_dir / "au_heatmap_data.csv")

    violin_aus = ANGER_AUS + EXTRA_AUS
    long_df = build_violin_long_frame(frame_data, violin_aus)
    plot_violins(long_df, violin_aus, output_dir / "au_violins.png")

    print(f"Wrote {output_dir / 'au_heatmap.png'}")
    print(f"Wrote {output_dir / 'au_violins.png'}")


if __name__ == "__main__":
    main()
