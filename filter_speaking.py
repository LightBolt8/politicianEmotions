"""Keep only face-crop frames where OpenFace AU25_c == 1 (lips part / speaking proxy)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def open_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def filter_video(
    video_path: Path,
    csv_path: Path | None = None,
    output_video: Path | None = None,
    output_csv: Path | None = None,
    *,
    require_success: bool = True,
) -> tuple[int, int]:
    """
    Write a speaking-only video/CSV keeping frames with AU25_c == 1.

    Returns (kept_frames, total_frames).
    """
    video_path = video_path.expanduser().resolve()
    csv_path = (csv_path or video_path.with_suffix(".csv")).expanduser().resolve()
    if output_video is None:
        output_video = video_path.with_name(f"{video_path.stem}_speaking{video_path.suffix}")
    if output_csv is None:
        output_csv = csv_path.with_name(f"{csv_path.stem}_speaking.csv")

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    if "AU25_c" not in df.columns:
        raise ValueError(f"AU25_c not found in {csv_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    n_video = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 5.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if len(df) != n_video:
        capture.release()
        raise ValueError(
            f"CSV/video length mismatch for {video_path.name}: "
            f"csv={len(df)} video={n_video}. Re-run OpenFace on this video first."
        )

    au25 = pd.to_numeric(df["AU25_c"], errors="coerce").fillna(0)
    keep = au25 == 1
    if require_success and "success" in df.columns:
        success = pd.to_numeric(df["success"], errors="coerce").fillna(0)
        keep &= success == 1

    keep_flags = keep.tolist()
    kept_df = df.loc[keep].copy()
    # Re-index frame/timestamp for the filtered clip.
    kept_df["frame"] = range(1, len(kept_df) + 1)
    if "timestamp" in kept_df.columns:
        kept_df["timestamp"] = [(i / fps) for i in range(len(kept_df))]

    writer = open_video_writer(output_video, fps, (width, height))
    kept = 0
    try:
        for idx in range(n_video):
            ret, frame = capture.read()
            if not ret:
                break
            if not keep_flags[idx]:
                continue
            writer.write(frame)
            kept += 1
    finally:
        capture.release()
        writer.release()

    kept_df.to_csv(output_csv, index=False)
    print(
        f"{video_path.name}: kept {kept}/{n_video} "
        f"({100 * kept / max(n_video, 1):.1f}%) -> {output_video.name}"
    )
    return kept, n_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discard non-speaking frames (AU25_c != 1) from face-crop videos."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        type=Path,
        help="Face-crop mp4 paths (OpenFace CSV must sit beside each video).",
    )
    parser.add_argument(
        "--keep-failed-tracks",
        action="store_true",
        help="Do not also require OpenFace success == 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for video in args.videos:
        filter_video(video, require_success=not args.keep_failed_tracks)


if __name__ == "__main__":
    main()
