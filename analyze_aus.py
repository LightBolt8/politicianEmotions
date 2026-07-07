"""Summarize anger AUs and plot per-candidate charts from OpenFace CSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANGER_AUS = ("AU04", "AU05", "AU07", "AU23")
ANGER_LABELS = {
    "AU04": "Brow lowerer",
    "AU05": "Upper lid raiser",
    "AU07": "Lid tightener",
    "AU23": "Lip tightener",
}
ANGER_COLOR = "#dc2626"
OTHER_COLOR = "#2563eb"

DEFAULT_DATA_DIR = Path("Exported")
DEFAULT_OUTPUT_ROOT = Path("analysis")


def openface_csv_for_video(video_path: Path) -> Path:
    """OpenFace CSV lives alongside the source video."""
    return video_path.parent / f"{video_path.stem}.csv"


def parse_video_name(video_path: Path) -> tuple[str, str]:
    if "_clean_" not in video_path.stem:
        raise ValueError(f"Expected Name_clean_YEAR.mp4, got: {video_path.name}")
    candidate, year = video_path.stem.rsplit("_clean_", 1)
    return candidate, year


def discover_exports(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*_clean_*.mp4"))


def run_openface_if_needed(
    video_path: Path,
    csv_path: Path,
    *,
    openface_bin: Path | None,
) -> None:
    if csv_path.is_file():
        return
    from run_openface import find_feature_extraction, run_openface_on_video

    binary = find_feature_extraction(openface_bin)
    openface_cwd = binary.parent.parent.parent if binary.parent.name == "bin" else None
    if openface_cwd is not None and not (openface_cwd / "model").is_dir():
        openface_cwd = None
    run_openface_on_video(
        binary,
        video_path.resolve(),
        csv_path.parent,
        openface_cwd=openface_cwd,
        tracked=False,
    )


def load_au_data(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    if "success" in df.columns:
        df = df[df["success"] == 1]
    au_cols = [col for col in df.columns if re.fullmatch(r"AU\d{2}_r", col)]
    if not au_cols:
        raise ValueError(f"No AU intensity columns found in {csv_path}")
    return df, au_cols


def build_summary(df: pd.DataFrame, au_cols: list[str]) -> pd.DataFrame:
    au_data = df[au_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "au": [col.replace("_r", "") for col in au_cols],
            "mean_intensity": au_data.mean().tolist(),
            "active_rate": (au_data > 0.1).mean().tolist(),
            "is_anger": [col.replace("_r", "") in ANGER_AUS for col in au_cols],
        }
    ).sort_values("mean_intensity", ascending=False)


def plot_anger_and_other(summary: pd.DataFrame, output_path: Path, *, title: str) -> None:
    anger = summary[summary["au"].isin(ANGER_AUS)].copy()
    anger["au"] = pd.Categorical(anger["au"], categories=list(ANGER_AUS), ordered=True)
    anger = anger.sort_values("au")
    other = summary[~summary["au"].isin(ANGER_AUS)].head(6)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    labels = [f"{row.au}\n{ANGER_LABELS.get(row.au, '')}" for row in anger.itertuples()]

    axes[0, 0].bar(labels, anger["mean_intensity"], color=ANGER_COLOR)
    axes[0, 0].set_title(f"{title} — anger AUs (mean intensity)")
    axes[0, 0].set_ylabel("Intensity (0–5)")

    axes[0, 1].bar(labels, anger["active_rate"] * 100, color=ANGER_COLOR)
    axes[0, 1].set_title(f"{title} — anger AUs (% active frames)")
    axes[0, 1].set_ylabel("Active (%)")

    axes[1, 0].bar(other["au"], other["mean_intensity"], color=OTHER_COLOR)
    axes[1, 0].set_title(f"{title} — other top AUs (mean intensity)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].bar(other["au"], other["active_rate"] * 100, color=OTHER_COLOR)
    axes[1, 1].set_title(f"{title} — other top AUs (% active frames)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_timeline(df: pd.DataFrame, summary: pd.DataFrame, output_path: Path, *, title: str) -> None:
    if "timestamp" in df.columns:
        time_axis = pd.to_numeric(df["timestamp"], errors="coerce")
    else:
        time_axis = pd.Series(np.arange(len(df)) / 5.0)

    anger_cols = [f"{au}_r" for au in ANGER_AUS if f"{au}_r" in df.columns]
    anger_data = df[anger_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    other_top = summary[~summary["au"].isin(ANGER_AUS)]["au"].head(3).tolist()
    other_cols = [f"{au}_r" for au in other_top if f"{au}_r" in df.columns]
    other_data = df[other_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for col in anger_cols:
        axes[0].plot(time_axis, anger_data[col], label=col.replace("_r", ""), linewidth=1.2)
    axes[0].set_title(f"{title} — anger AUs over time")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[0].grid(True, alpha=0.25)

    for col in other_cols:
        axes[1].plot(time_axis, other_data[col], label=col.replace("_r", ""), linewidth=1.2)
    axes[1].set_title(f"{title} — contextual AUs over time")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend(loc="upper right", ncol=2, fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze_video(
    video_path: Path,
    *,
    output_root: Path,
    openface_bin: Path | None,
    run_openface: bool,
) -> Path:
    candidate, year = parse_video_name(video_path)
    csv_path = openface_csv_for_video(video_path)
    if run_openface:
        run_openface_if_needed(video_path, csv_path, openface_bin=openface_bin)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"OpenFace CSV not found: {csv_path}. Run run_openface.py first or pass --run-openface."
        )

    work_dir = output_root / year / candidate
    plots_dir = work_dir / "plots"
    title = f"{candidate} ({year})"

    df, au_cols = load_au_data(csv_path)
    summary = build_summary(df, au_cols)
    anger_summary = summary[summary["au"].isin(ANGER_AUS)].copy()

    plot_anger_and_other(summary, plots_dir / "anger_and_other.png", title=title)
    plot_timeline(df, summary, plots_dir / "timeline.png", title=title)
    summary.to_csv(work_dir / "au_summary.csv", index=False)
    anger_summary.to_csv(work_dir / "anger_aus.csv", index=False)

    anger_idx = anger_summary["mean_intensity"].mean()
    print(f"[{year}/{candidate}] Frames: {len(df)} | Anger index: {anger_idx:.2f}")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot anger-focused AU analysis from OpenFace CSVs.")
    parser.add_argument("--source", type=Path, help="Single exported face-crop video.")
    parser.add_argument("--year", type=str, help="Debate year (optional if --source name includes it).")
    parser.add_argument("--candidate", type=str, help="Candidate name (optional if --source name includes it).")
    parser.add_argument("--all", action="store_true", help="Analyze every Exported/*_clean_*.mp4.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--exported-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-openface",
        action="store_true",
        help="Run OpenFace if the CSV is missing (default: require existing CSV).",
    )
    parser.add_argument(
        "--openface-bin",
        type=Path,
        default=Path.home() / "OpenFace/build/bin/FeatureExtraction",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Also write cross-candidate heatmap/violin plots (on by default with --all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = (args.exported_dir or args.data_dir).expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.all:
        videos = discover_exports(data_dir)
        if not videos:
            raise FileNotFoundError(f"No *_clean_*.mp4 files under {data_dir}")
        failures: list[str] = []
        for video in videos:
            try:
                analyze_video(
                    video,
                    output_root=output_root,
                    openface_bin=args.openface_bin,
                    run_openface=args.run_openface,
                )
            except FileNotFoundError as exc:
                failures.append(f"{video.name}: {exc}")
        if failures:
            print("Skipped (missing OpenFace CSV):")
            for msg in failures:
                print(f"  {msg}")
        from plot_au_comparison import run_comparison

        run_comparison(data_dir, output_root / "comparison")
        return

    if not args.source:
        raise SystemExit("Pass --source or use --all.")

    video = args.source.expanduser().resolve()
    analyze_video(
        video,
        output_root=output_root,
        openface_bin=args.openface_bin,
        run_openface=args.run_openface,
    )
    if args.comparison:
        from plot_au_comparison import run_comparison

        run_comparison(data_dir, output_root / "comparison")


if __name__ == "__main__":
    main()
