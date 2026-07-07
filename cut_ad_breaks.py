"""Remove ad-break segments from exported face-crop videos."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

# (start_seconds, end_seconds) — segments to remove
AD_BREAKS: dict[str, list[tuple[float, float]]] = {
    "Trump_clean_2024.mp4": [(3254, 3540), (5012, 5288)],
    "Harris_clean_2024.mp4": [(3036, 3314), (4748, 5024)],
}


def keep_segments(
    duration: float, cuts: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return timestamp ranges to keep, given cuts to remove."""
    cuts = sorted(cuts)
    segments: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in cuts:
        if start > cursor:
            segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        segments.append((cursor, duration))
    return segments


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def cut_video(source: Path, output: Path, cuts: list[tuple[float, float]]) -> None:
    duration = probe_duration(source)
    segments = keep_segments(duration, cuts)
    if not segments:
        raise ValueError(f"No segments left after cuts: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    filter_parts: list[str] = []
    labels: list[str] = []
    for idx, (start, end) in enumerate(segments):
        label = f"v{idx}"
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")

    filter_graph = ";".join(filter_parts)
    filter_graph += f";{''.join(labels)}concat=n={len(labels)}:v=1:a=0[outv]"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        filter_graph,
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-an",
        str(output),
    ]
    subprocess.run(command, check=True)
    removed = sum(end - start for start, end in cuts)
    kept = probe_duration(output)
    print(
        f"{source.name}: {duration:.1f}s -> {kept:.1f}s "
        f"(removed {removed:.1f}s across {len(cuts)} ad breaks)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut ad breaks from exported videos.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("Exported/Trump vs Harris"),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_noads",
        help="Suffix for cleaned output files (before .mp4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()

    for filename, cuts in AD_BREAKS.items():
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing source video: {source}")
        stem = source.stem
        if stem.endswith(args.suffix):
            stem = stem[: -len(args.suffix)]
        output = source_dir / f"{stem}{args.suffix}.mp4"
        cut_video(source, output, cuts)


if __name__ == "__main__":
    main()
