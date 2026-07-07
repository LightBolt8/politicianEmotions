"""Run OpenFace on a face-crop clip and plot anger AUs with contextual AUs."""

from __future__ import annotations

import argparse
import re
import subprocess
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


def extract_clip(source: Path, output: Path, *, start_seconds: float, duration_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(source),
        "-t",
        str(duration_seconds),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-an",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}"
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no output: {output}")


def run_openface(binary: Path, video: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    video = video.resolve()
    out_dir = out_dir.resolve()
    command = [
        str(binary),
        "-f",
        str(video),
        "-out_dir",
        str(out_dir),
        "-2Dfp",
        "-pose",
        "-aus",
    ]
    subprocess.run(command, check=True, cwd=binary.parent.parent.parent)
    csv_path = out_dir / f"{video.stem}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")
    return csv_path


def load_au_data(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    if "success" in df.columns:
        df = df[df["success"] == 1]
    au_cols = [col for col in df.columns if re.fullmatch(r"AU\d{2}_r", col)]
    if not au_cols:
        raise ValueError("No AU intensity columns found in OpenFace CSV")
    return df, au_cols


def build_summary(df: pd.DataFrame, au_cols: list[str]) -> pd.DataFrame:
    au_data = df[au_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mean_intensity = au_data.mean()
    active_rate = (au_data > 0.1).mean()
    return pd.DataFrame(
        {
            "au": [col.replace("_r", "") for col in au_cols],
            "mean_intensity": [mean_intensity[col] for col in au_cols],
            "active_rate": [active_rate[col] for col in au_cols],
            "is_anger": [col.replace("_r", "") in ANGER_AUS for col in au_cols],
        }
    ).sort_values("mean_intensity", ascending=False)


def plot_anger_and_other(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
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
    axes[1, 0].set_ylabel("Intensity (0–5)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].bar(other["au"], other["active_rate"] * 100, color=OTHER_COLOR)
    axes[1, 1].set_title(f"{title} — other top AUs (% active frames)")
    axes[1, 1].set_ylabel("Active (%)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_timeline(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    if "timestamp" in df.columns:
        time_axis = pd.to_numeric(df["timestamp"], errors="coerce")
    elif "frame" in df.columns:
        time_axis = pd.to_numeric(df["frame"], errors="coerce") / 5.0
    else:
        time_axis = pd.Series(np.arange(len(df)) / 5.0)

    anger_cols = [f"{au}_r" for au in ANGER_AUS if f"{au}_r" in df.columns]
    anger_data = df[anger_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    other_top = (
        summary[~summary["au"].isin(ANGER_AUS)]["au"].head(3).tolist()
    )
    other_cols = [f"{au}_r" for au in other_top if f"{au}_r" in df.columns]
    other_data = df[other_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    for col in anger_cols:
        axes[0].plot(time_axis, anger_data[col], label=col.replace("_r", ""), linewidth=1.2)
    axes[0].set_title(f"{title} — anger AUs over time")
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[0].grid(True, alpha=0.25)

    for col in other_cols:
        axes[1].plot(time_axis, other_data[col], label=col.replace("_r", ""), linewidth=1.2)
    axes[1].set_title(f"{title} — contextual AUs over time")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Intensity")
    axes[1].legend(loc="upper right", ncol=2, fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenFace AU analysis focused on anger AUs.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument(
        "--openface-bin",
        type=Path,
        default=Path.home() / "OpenFace/build/bin/FeatureExtraction",
    )
    parser.add_argument("--output-root", type=Path, default=Path("analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.output_root.expanduser() / args.year / args.candidate
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_path = work_dir / "clip.mp4"
    openface_dir = work_dir / "openface"
    plots_dir = work_dir / "plots"
    title = f"{args.candidate} ({args.year})"

    if not args.openface_bin.is_file():
        raise FileNotFoundError(f"OpenFace binary not found: {args.openface_bin}")

    print(f"[{args.year}/{args.candidate}] Extracting {args.duration_seconds}s clip...")
    extract_clip(
        args.source.expanduser().resolve(),
        clip_path,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
    )

    print(f"[{args.year}/{args.candidate}] Running OpenFace...")
    csv_path = run_openface(args.openface_bin.resolve(), clip_path, openface_dir)

    df, au_cols = load_au_data(csv_path)
    summary = build_summary(df, au_cols)

    anger_summary = summary[summary["au"].isin(ANGER_AUS)].copy()
    anger_summary["anger_index"] = anger_summary["mean_intensity"].mean()

    plot_anger_and_other(summary, plots_dir / "anger_and_other.png", title=title)
    plot_timeline(df, summary, plots_dir / "timeline.png", title=title)
    summary.to_csv(work_dir / "au_summary.csv", index=False)
    anger_summary.to_csv(work_dir / "anger_aus.csv", index=False)

    anger_idx = anger_summary["mean_intensity"].mean()
    print(f"[{args.year}/{args.candidate}] Frames: {len(df)} | Anger index: {anger_idx:.2f}")
    print(f"[{args.year}/{args.candidate}] Top anger AU: {anger_summary.sort_values('mean_intensity', ascending=False).iloc[0]['au']}")


if __name__ == "__main__":
    main()
