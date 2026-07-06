"""Run OpenFace FeatureExtraction on preprocessed candidate face videos."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_EXPORTED_DIR = Path("Exported")
DEFAULT_OUTPUT_DIR = Path("OpenFaceResults")

# CSV + features only; skip tracked video / aligned images to save disk space.
OPENFACE_FEATURE_FLAGS = ("-2Dfp", "-3Dfp", "-pose", "-aus", "-gaze")


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


def discover_videos(exported_dir: Path, videos: list[Path] | None) -> list[Path]:
    """Return processed MP4 files to analyze."""
    if videos:
        found = [path.expanduser().resolve() for path in videos]
        missing = [path for path in found if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Video not found: {missing[0]}")
        return found

    if not exported_dir.is_dir():
        raise FileNotFoundError(f"Exported directory not found: {exported_dir}")

    found = sorted(exported_dir.rglob("*.mp4"))
    if not found:
        raise FileNotFoundError(f"No .mp4 files found under {exported_dir}")
    return found


def output_dir_for_video(
    video_path: Path, exported_dir: Path, output_root: Path
) -> Path:
    """Mirror Exported/<debate>/ under OpenFaceResults/<debate>/."""
    try:
        relative_parent = video_path.parent.relative_to(exported_dir.resolve())
    except ValueError:
        relative_parent = Path(video_path.parent.name)
    return output_root / relative_parent / video_path.stem


def csv_already_exists(out_dir: Path, video_path: Path) -> bool:
    return (out_dir / f"{video_path.stem}.csv").is_file()


def run_openface_on_video(
    binary: Path,
    video_path: Path,
    out_dir: Path,
    *,
    openface_cwd: Path | None,
) -> None:
    """Run FeatureExtraction for a single preprocessed face video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "-f",
        str(video_path),
        "-out_dir",
        str(out_dir),
        *OPENFACE_FEATURE_FLAGS,
    ]
    print(f"Running OpenFace on {video_path.name} -> {out_dir}")
    subprocess.run(
        command,
        check=True,
        cwd=openface_cwd,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenFace FeatureExtraction on preprocessed candidate videos "
            "from preprocessing_data.py."
        )
    )
    parser.add_argument(
        "--exported-dir",
        type=Path,
        default=DEFAULT_EXPORTED_DIR,
        help="Directory containing preprocessed videos (default: Exported).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for OpenFace CSV output (default: OpenFaceResults).",
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
    binary, openface_cwd = resolve_binary(args)
    exported_dir = args.exported_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    videos = discover_videos(exported_dir, args.videos or None)

    print(f"Using OpenFace binary: {binary}")
    print(f"Found {len(videos)} video(s) to process")

    failures: list[str] = []
    for video_path in videos:
        out_dir = output_dir_for_video(video_path, exported_dir, output_root)
        if not args.force and csv_already_exists(out_dir, video_path):
            print(f"Skipping {video_path.name} (CSV already exists)")
            continue
        try:
            run_openface_on_video(
                binary,
                video_path,
                out_dir,
                openface_cwd=openface_cwd,
            )
        except subprocess.CalledProcessError as exc:
            failures.append(f"{video_path.name} (exit code {exc.returncode})")

    if failures:
        print("OpenFace failed for:", ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
