"""Export face-crop clips + combined OpenFace CSV for fixed debate-clock windows.

Windows (source video time): 10–12, 40–42, and 70–72 minutes.
Outputs go under Exported/time_windows/.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "Exported" / "time_windows"
VIDEOS_DIR = OUT_ROOT / "videos"
SOURCE_DIR = OUT_ROOT / "source_clips"

WINDOWS = (
    ("10-12min", 10 * 60, 2 * 60),
    ("40-42min", 40 * 60, 2 * 60),
    ("70-72min", 70 * 60, 2 * 60),
)

# (year_label, dataset_stem, candidate_A_name, candidate_B_name, start_offset_sec)
YEARS = (
    ("2004", "2004", "Bush", "Kerry", 0.0),
    ("2008", "2008", "Obama", "McCain", 0.0),
    ("2012", "2012", "Obama", "Romney", 0.0),
    ("2016", "2016", "Trump", "Clinton", 0.0),
    ("2020", "2020", "Trump", "Biden", 0.0),
    ("2024b", "2024b", "Trump", "Biden", 11.0),  # skip intro bumper used in full run
    ("2024k", "2024k", "Trump", "Harris", 0.0),
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def stem_for(year: str, candidate: str, window: str) -> str:
    return f"{year}_{candidate}_{window}"


def export_source_clip(dataset: Path, start: float, duration: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 0:
        print(f"skip existing source clip {out.name}")
        return
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(dataset),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            str(out),
        ]
    )


def preprocess_window(
    year: str,
    dataset_stem: str,
    name_a: str,
    name_b: str,
    start_offset: float,
    window: str,
    start_min_sec: float,
    duration: float,
    *,
    force: bool,
) -> tuple[Path, Path]:
    start = start_offset + start_min_sec
    work = OUT_ROOT / "work" / year / window
    out_a = work / f"{name_a}.mp4"
    out_b = work / f"{name_b}.mp4"
    final_a = VIDEOS_DIR / f"{stem_for(year, name_a, window)}.mp4"
    final_b = VIDEOS_DIR / f"{stem_for(year, name_b, window)}.mp4"

    if (
        not force
        and final_a.is_file()
        and final_b.is_file()
        and final_a.stat().st_size > 0
        and final_b.stat().st_size > 0
    ):
        print(f"skip preprocess {year} {window} (videos exist)")
        return final_a, final_b

    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "preprocessing_data.py",
        "--input-video",
        f"Dataset/{dataset_stem}.mp4",
        "--candidate-a",
        f"refs/{dataset_stem}/candidate_A.jpg",
        "--candidate-b",
        f"refs/{dataset_stem}/candidate_B.jpg",
        "--output-a",
        str(out_a),
        "--output-b",
        str(out_b),
        "--start-seconds",
        f"{start:.3f}",
        "--max-seconds",
        f"{duration:.3f}",
        "--threshold",
        "0.6",
        "--checkpoint-every-frames",
        "100000",
    ]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"WARN preprocess failed for {year} {window} (exit {result.returncode})")
        return final_a, final_b
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    if out_a.is_file():
        shutil.copy2(out_a, final_a)
    if out_b.is_file():
        shutil.copy2(out_b, final_b)
    return final_a, final_b


def openface_videos(videos: list[Path], *, force: bool) -> None:
    need = []
    for v in videos:
        csv = v.with_suffix(".csv")
        if force or not csv.is_file() or csv.stat().st_size == 0:
            need.append(v)
        else:
            print(f"skip OpenFace {v.name}")
    if not need:
        return
    cmd = [sys.executable, "run_openface.py"]
    if force:
        cmd.append("--force")
    cmd.extend(str(v) for v in need)
    run(cmd)


def build_combined_csv(paths: list[tuple[str, str, str, float, Path]]) -> Path:
    """paths entries: year, candidate, window, source_start_sec, video_path."""
    frames: list[pd.DataFrame] = []
    for year, candidate, window, source_start, video in paths:
        csv_path = video.with_suffix(".csv")
        if not csv_path.is_file():
            print(f"WARN missing CSV for {video.name}")
            continue
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        meta = pd.DataFrame(
            {
                "year": year,
                "candidate": candidate,
                "window": window,
                "source_start_sec": source_start,
                "source_end_sec": source_start + 120.0,
                "clip_file": video.name,
            },
            index=df.index,
        )
        frames.append(pd.concat([meta, df], axis=1))

    if not frames:
        raise RuntimeError("No OpenFace CSVs found to combine")
    combined = pd.concat(frames, ignore_index=True)
    out = OUT_ROOT / "all_windows_openface.csv"
    combined.to_csv(out, index=False)
    print(f"Wrote {out} ({len(combined)} rows, {combined['clip_file'].nunique()} clips)")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Re-run preprocess/OpenFace even if outputs exist.")
    p.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Only OpenFace + combine existing face-crop clips.",
    )
    p.add_argument(
        "--skip-source-clips",
        action="store_true",
        help="Do not export raw debate ffmpeg clips.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    catalog: list[tuple[str, str, str, float, Path]] = []
    all_videos: list[Path] = []

    for year, dataset_stem, name_a, name_b, start_offset in YEARS:
        dataset = ROOT / "Dataset" / f"{dataset_stem}.mp4"
        if not dataset.is_file():
            print(f"WARN missing dataset {dataset}")
            continue
        for window, start_min_sec, duration in WINDOWS:
            source_start = start_offset + start_min_sec
            if not args.skip_source_clips:
                export_source_clip(
                    dataset,
                    source_start,
                    duration,
                    SOURCE_DIR / f"{year}_{window}_source.mp4",
                )
            if args.skip_preprocess:
                va = VIDEOS_DIR / f"{stem_for(year, name_a, window)}.mp4"
                vb = VIDEOS_DIR / f"{stem_for(year, name_b, window)}.mp4"
            else:
                va, vb = preprocess_window(
                    year,
                    dataset_stem,
                    name_a,
                    name_b,
                    start_offset,
                    window,
                    start_min_sec,
                    duration,
                    force=args.force,
                )
            for name, path in ((name_a, va), (name_b, vb)):
                if path.is_file() and path.stat().st_size > 0:
                    catalog.append((year, name, window, source_start, path))
                    all_videos.append(path)
                else:
                    print(f"WARN missing face clip {path}")

    openface_videos(all_videos, force=args.force)
    build_combined_csv(catalog)
    print(f"Done. Folder: {OUT_ROOT}")


if __name__ == "__main__":
    main()
