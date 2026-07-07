"""Run OpenFace FeatureExtraction on preprocessed candidate face videos."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("Exported")

# Minimal OpenFace output: pose + AUs + gaze (no 2D/3D landmark columns).
OPENFACE_FEATURE_FLAGS = ("-pose", "-aus", "-gaze")

CORE_CSV_COLUMNS: tuple[str, ...] = (
    "frame",
    "face_id",
    "timestamp",
    "confidence",
    "success",
    "gaze_angle_x",
    "gaze_angle_y",
    "pose_Tx",
    "pose_Ty",
    "pose_Tz",
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
)


def openface_flags(*, tracked: bool = False) -> tuple[str, ...]:
    flags = list(OPENFACE_FEATURE_FLAGS)
    if tracked:
        flags.append("-tracked")
    return tuple(flags)


def au_columns(columns: list[str]) -> list[str]:
    return sorted(
        [col for col in columns if re.fullmatch(r"AU\d{2}_[rc]", col)],
        key=lambda col: (int(col[2:4]), col[-1]),
    )


def slim_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip() for col in df.columns]
    keep = [col for col in CORE_CSV_COLUMNS if col in df.columns]
    keep.extend(au_columns(list(df.columns)))
    if not keep:
        raise ValueError("No recognized OpenFace columns found")
    return df[keep]


def slim_openface_csv(csv_path: Path) -> tuple[int, int]:
    """Rewrite an OpenFace CSV keeping only core metadata, gaze, pose, and AUs."""
    before_cols = len(pd.read_csv(csv_path, nrows=0).columns)
    df = pd.read_csv(csv_path)
    slim = slim_csv_columns(df)
    slim.to_csv(csv_path, index=False)
    return before_cols, len(slim.columns)


def discover_openface_csvs(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*_clean_*.csv")
        if path.parent.name == path.stem and path.is_file()
    )


def slim_all_csvs(data_dir: Path) -> None:
    csv_paths = discover_openface_csvs(data_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No OpenFace CSVs found under {data_dir}")
    for csv_path in csv_paths:
        before, after = slim_openface_csv(csv_path)
        print(f"Slimmed {csv_path.name}: {before} -> {after} columns")


def find_feature_extraction(explicit: Path | None) -> Path:
    """Locate the OpenFace FeatureExtraction binary."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_bin = os.environ.get("OPENFACE_BIN")
    if env_bin:
        candidates.append(Path(env_bin).expanduser())
    candidates.extend(
        [
            Path.home() / "OpenFace" / "build" / "bin" / "FeatureExtraction",
            Path("/opt/OpenFace/build/bin/FeatureExtraction"),
            Path("/usr/local/bin/FeatureExtraction"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise FileNotFoundError(
        "OpenFace FeatureExtraction binary not found. Install OpenFace from "
        "https://github.com/TadasBaltrusaitis/OpenFace and pass "
        "--openface-bin /path/to/FeatureExtraction or set OPENFACE_BIN."
    )


def discover_videos(data_dir: Path, videos: list[Path] | None) -> list[Path]:
    """Return processed MP4 files to analyze."""
    if videos:
        found = [path.expanduser().resolve() for path in videos]
        missing = [path for path in found if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Video not found: {missing[0]}")
        return found

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    found = sorted(data_dir.rglob("*.mp4"))
    if not found:
        raise FileNotFoundError(f"No .mp4 files found under {data_dir}")
    return found


def output_dir_for_video(video_path: Path) -> Path:
    """Write OpenFace outputs next to the source video."""
    return video_path.parent


def csv_already_exists(out_dir: Path, video_path: Path) -> bool:
    return (out_dir / f"{video_path.stem}.csv").is_file()


def tracked_video_exists(out_dir: Path, video_path: Path) -> bool:
    return (out_dir / f"{video_path.stem}.avi").is_file()


def should_skip_video(
    out_dir: Path,
    video_path: Path,
    *,
    force: bool,
    tracked: bool,
) -> bool:
    if force:
        return False
    if not csv_already_exists(out_dir, video_path):
        return False
    if tracked and not tracked_video_exists(out_dir, video_path):
        return False
    return True


def run_openface_on_video(
    binary: Path,
    video_path: Path,
    out_dir: Path,
    *,
    openface_cwd: Path | None,
    tracked: bool = False,
) -> None:
    """Run FeatureExtraction for a single preprocessed face video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "-f",
        str(video_path),
        "-out_dir",
        str(out_dir),
        *openface_flags(tracked=tracked),
    ]
    print(f"Running OpenFace on {video_path.name} -> {out_dir}")
    if tracked:
        print(f"  Tracked video will be written to {out_dir / (video_path.stem + '.avi')}")
    subprocess.run(
        command,
        check=True,
        cwd=openface_cwd,
    )
    csv_path = out_dir / f"{video_path.stem}.csv"
    if csv_path.is_file():
        before, after = slim_openface_csv(csv_path)
        print(f"  Slimmed CSV: {before} -> {after} columns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenFace FeatureExtraction on preprocessed candidate videos "
            "from preprocessing_data.py."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing preprocessed videos (default: Exported).",
    )
    parser.add_argument(
        "--exported-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--openface-bin",
        type=Path,
        default=None,
        help="Path to OpenFace FeatureExtraction binary.",
    )
    parser.add_argument(
        "--openface-dir",
        type=Path,
        default=None,
        help="OpenFace install directory (uses <dir>/build/bin/FeatureExtraction).",
    )
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Specific video files to process (default: all under --exported-dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if the CSV output already exists.",
    )
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="Also export a tracked .avi with landmarks and AU overlay.",
    )
    parser.add_argument(
        "--slim-csvs",
        action="store_true",
        help="Trim existing OpenFace CSVs under --data-dir (no FeatureExtraction run).",
    )
    return parser.parse_args()


def resolve_binary(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.openface_bin is not None:
        binary = find_feature_extraction(args.openface_bin)
    elif args.openface_dir is not None:
        binary = find_feature_extraction(
            args.openface_dir.expanduser() / "build" / "bin" / "FeatureExtraction"
        )
    else:
        binary = find_feature_extraction(None)

    openface_cwd = binary.parent.parent.parent if binary.parent.name == "bin" else None
    if openface_cwd is not None and not (openface_cwd / "model").is_dir():
        openface_cwd = None
    return binary, openface_cwd


def main() -> None:
    args = parse_args()
    data_dir = (args.exported_dir or args.data_dir).expanduser().resolve()

    if args.slim_csvs:
        slim_all_csvs(data_dir)
        return

    binary, openface_cwd = resolve_binary(args)
    videos = discover_videos(data_dir, args.videos or None)

    print(f"Using OpenFace binary: {binary}")
    print(f"Found {len(videos)} video(s) to process")

    failures: list[str] = []
    for video_path in videos:
        out_dir = output_dir_for_video(video_path)
        if should_skip_video(out_dir, video_path, force=args.force, tracked=args.tracked):
            print(f"Skipping {video_path.name} (outputs already exist)")
            continue
        try:
            run_openface_on_video(
                binary,
                video_path,
                out_dir,
                openface_cwd=openface_cwd,
                tracked=args.tracked,
            )
        except subprocess.CalledProcessError as exc:
            failures.append(f"{video_path.name} (exit code {exc.returncode})")

    if failures:
        print("OpenFace failed for:", ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
