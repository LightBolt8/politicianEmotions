"""Cross-candidate AU heatmap and violin plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANGER_AUS = ("AU04", "AU05", "AU07")
ANGER_LABELS = {
    "AU04": "Brow lowerer",
    "AU05": "Upper lid raiser",
    "AU07": "Lid tightener",
}
EXTRA_AUS = ("AU12", "AU14", "AU17", "AU26")
EXTRA_LABELS = {
    "AU12": "Lip corner puller",
    "AU14": "Dimpler",
    "AU17": "Chin raiser",
    "AU26": "Jaw drop",
}

# Default comparison paths (prefer speaking CSVs after 2008).
DEFAULT_CANDIDATES: list[tuple[str, str]] = [
    ("Trump 2016", "2016/Trump_clean_2016/Trump_clean_2016_speaking.csv"),
    ("Clinton 2016", "2016/Clinton_clean_2016/Clinton_clean_2016_speaking.csv"),
    ("Trump 2024", "2024k/Trump_clean_2024/Trump_clean_2024_speaking.csv"),
    ("Harris 2024", "2024k/Harris_clean_2024/Harris_clean_2024_speaking.csv"),
]

PALETTE = {
    "Trump 2016": "#dc2626",
    "Clinton 2016": "#2563eb",
    "Trump 2024": "#f97316",
    "Harris 2024": "#7c3aed",
    "Biden 2020": "#16a34a",
    "Trump 2020": "#ea580c",
}


def csv_path_for(data_root: Path, relative: str) -> Path:
    return data_root / relative


def discover_all_candidates(data_root: Path) -> list[tuple[str, Path]]:
    """Prefer OpenFace CSVs from speaking videos after 2008; full CSVs otherwise."""
    found: list[tuple[str, Path, tuple[int, str]]] = []
    for csv_path in sorted(data_root.rglob("*_clean_*.csv")):
        stem = csv_path.stem
        if stem.endswith("_nonspeaking"):
            continue
        speaking = stem.endswith("_speaking")
        base_stem = stem[: -len("_speaking")] if speaking else stem
        if "_clean_" not in base_stem:
            continue
        # Require CSV to live in the candidate folder named after the non-speaking stem.
        if csv_path.parent.name != base_stem:
            continue
        candidate, year = base_stem.rsplit("_clean_", 1)
        year_key = int(re.match(r"(\d+)", year).group(1)) if re.match(r"(\d+)", year) else 0
        # No AU25 speaking filter for 2008 and earlier.
        if year_key <= 2008 and speaking:
            continue
        # After 2008, require speaking OpenFace CSV when the speaking video exists.
        if year_key > 2008 and not speaking:
            speaking_video = csv_path.with_name(f"{base_stem}_speaking.mp4")
            speaking_csv = csv_path.with_name(f"{base_stem}_speaking.csv")
            if speaking_video.is_file() or speaking_csv.is_file():
                continue
        label = f"{candidate} {year}"
        found.append((label, csv_path, (year_key, candidate)))

    best: dict[str, tuple[Path, tuple[int, str]]] = {}
    for label, path, sort_key in found:
        if label not in best or path.stat().st_size > best[label][0].stat().st_size:
            best[label] = (path, sort_key)

    return [
        (label, path)
        for label, (path, _) in sorted(best.items(), key=lambda x: x[1][1])
    ]


def resolve_candidates(data_root: Path, *, include_all: bool) -> list[tuple[str, Path]]:
    if include_all:
        return discover_all_candidates(data_root)

    candidates: list[tuple[str, Path]] = []
    missing: list[str] = []
    for label, relative in DEFAULT_CANDIDATES:
        path = csv_path_for(data_root, relative)
        if not path.is_file():
            missing.append(f"{label} ({path})")
            continue
        candidates.append((label, path))
    if not candidates:
        raise FileNotFoundError(
            "No comparison CSVs found. Missing:\n  " + "\n  ".join(missing)
        )
    if missing:
        print("Skipping missing candidates:")
        for msg in missing:
            print(f"  {msg}")
    return candidates


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
    rows = {label: df.mean() for label, df in frame_data.items()}
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
    # Single-line titles avoid colliding with the figure title.
    if au in ANGER_LABELS:
        return f"{au} — {ANGER_LABELS[au]}"
    if au in EXTRA_LABELS:
        return f"{au} — {EXTRA_LABELS[au]}"
    return au


def build_au_summary_table(
    frame_data: dict[str, pd.DataFrame],
    aus: tuple[str, ...] = ANGER_AUS,
) -> pd.DataFrame:
    """Mean / SD / n frames for each candidate × AU (spreadsheet-ready)."""
    rows: list[dict[str, object]] = []
    for label, df in frame_data.items():
        row: dict[str, object] = {"candidate": label, "n_frames": len(df)}
        for au in aus:
            col = f"{au}_r"
            if col not in df.columns:
                row[f"{au}_mean"] = float("nan")
                row[f"{au}_sd"] = float("nan")
                continue
            series = df[col]
            row[f"{au}_mean"] = float(series.mean())
            row[f"{au}_sd"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def export_au_spreadsheet(summary: pd.DataFrame, output_path: Path) -> Path:
    """Write AU4/5/7 summary as .xlsx (falls back to .csv if openpyxl missing)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary.to_excel(output_path, index=False, sheet_name="AU4_5_7")
        return output_path
    except ImportError:
        csv_path = output_path.with_suffix(".csv")
        summary.to_csv(csv_path, index=False)
        return csv_path


def plot_au_summary_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Render AU4/5/7 means (± SD) as a table image."""
    display = pd.DataFrame(
        {
            "Candidate": summary["candidate"],
            "n": summary["n_frames"].map(lambda n: f"{int(n):,}"),
            "AU04": [
                f"{m:.3f} ± {s:.3f}"
                for m, s in zip(summary["AU04_mean"], summary["AU04_sd"])
            ],
            "AU05": [
                f"{m:.3f} ± {s:.3f}"
                for m, s in zip(summary["AU05_mean"], summary["AU05_sd"])
            ],
            "AU07": [
                f"{m:.3f} ± {s:.3f}"
                for m, s in zip(summary["AU07_mean"], summary["AU07_sd"])
            ],
        }
    )

    n_rows = len(display)
    fig_h = max(3.2, 0.42 * n_rows + 1.4)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis("off")
    ax.set_title(
        "Mean AU intensity (± SD) — AU04 / AU05 / AU07",
        fontsize=13,
        pad=12,
        fontweight="bold",
    )

    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=list(display.columns),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d4d8")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#27272a")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.08)
        else:
            cell.set_facecolor("#fafafa" if row % 2 else "white")
        if col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.04

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_violins(
    long_df: pd.DataFrame,
    aus: tuple[str, ...],
    candidate_order: list[str],
    output_path: Path,
) -> None:
    colors = sns.color_palette("husl", n_colors=len(candidate_order))
    palette = {
        name: PALETTE.get(name, colors[i])
        for i, name in enumerate(candidate_order)
    }

    ncols = min(3, len(aus))
    nrows = int(np.ceil(len(aus) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.2 * nrows),
        layout="constrained",
    )
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
        ax.set_title(au_panel_label(au), fontsize=10, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, min(5.0, panel["intensity"].quantile(0.995) * 1.15 + 0.1))
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    for ax in axes_flat[len(aus) :]:
        ax.axis("off")

    fig.suptitle("AU intensity distributions by candidate", fontsize=14)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.08, hspace=0.12, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cross-candidate AU comparisons.")
    parser.add_argument("--data-dir", type=Path, default=Path("Exported"))
    parser.add_argument(
        "--openface-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Exported/comparison"))
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include all candidates under Exported/ (default: 2016 + 2024 only).",
    )
    return parser.parse_args(argv)


def run_comparison(
    data_root: Path | None = None,
    output_dir: Path | None = None,
    *,
    include_all: bool = False,
) -> None:
    data_root = (data_root or Path("Exported")).expanduser().resolve()
    output_dir = (output_dir or Path("Exported/comparison")).expanduser().resolve()

    candidates = resolve_candidates(data_root, include_all=include_all)
    frame_data = {label: load_au_frame_data(path) for label, path in candidates}
    for label, path in candidates:
        print(f"Loaded {label}: {path.name} ({len(frame_data[label])} frames)")

    mean_matrix = build_mean_matrix(frame_data)
    # Heatmap focuses on aggression-related AUs (AU4/5/7).
    anger_cols = [au for au in ANGER_AUS if au in mean_matrix.columns]
    anger_matrix = mean_matrix[anger_cols]
    plot_heatmap(anger_matrix, output_dir / "au_heatmap.png")
    anger_matrix.to_csv(output_dir / "au_heatmap_data.csv")

    summary = build_au_summary_table(frame_data, ANGER_AUS)
    summary_path = export_au_spreadsheet(summary, output_dir / "au4_5_7_summary.xlsx")
    summary.to_csv(output_dir / "au4_5_7_summary.csv", index=False)
    plot_au_summary_table(summary, output_dir / "au4_5_7_summary.png")

    violin_aus = ANGER_AUS
    long_df = build_violin_long_frame(frame_data, violin_aus)
    candidate_order = [label for label, _ in candidates]
    plot_violins(long_df, violin_aus, candidate_order, output_dir / "au_violins.png")

    print(f"Wrote {output_dir / 'au_heatmap.png'}")
    print(f"Wrote {output_dir / 'au_violins.png'}")
    print(f"Wrote {output_dir / 'au4_5_7_summary.png'}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {output_dir / 'au4_5_7_summary.csv'}")


def main() -> None:
    args = parse_args()
    data_dir = args.openface_dir or args.data_dir
    run_comparison(data_dir, args.output_dir, include_all=args.all)


if __name__ == "__main__":
    main()
