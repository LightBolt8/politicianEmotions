"""Build speaking/nonspeaking face-crop videos from full-video OpenFace AU25_c.

Uses the full clean OpenFace CSV only as a frame mask. Analysis OpenFace CSVs for
speaking clips should be produced by running OpenFace on *_speaking.mp4.
"""

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
    output_deleted_video: Path | None = None,
    *,
    require_success: bool = True,
    write_deleted: bool = True,
) -> tuple[int, int, int]:
    """
    Write speaking-only (and optionally nonspeaking) videos using AU25_c == 1.

    Does not write a speaking CSV — run OpenFace on the speaking video for that.

    Returns (kept_frames, deleted_frames, total_frames).
    """
    video_path = video_path.expanduser().resolve()
    csv_path = (csv_path or video_path.with_suffix(".csv")).expanduser().resolve()
    if output_video is None:
        output_video = video_path.with_name(f"{video_path.stem}_speaking{video_path.suffix}")
    if output_deleted_video is None:
        output_deleted_video = video_path.with_name(
            f"{video_path.stem}_nonspeaking{video_path.suffix}"
        )

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
    writer = open_video_writer(output_video, fps, (width, height))
    deleted_writer = (
        open_video_writer(output_deleted_video, fps, (width, height))
        if write_deleted
        else None
    )
    kept = 0
    deleted = 0
    try:
        for idx in range(n_video):
            ret, frame = capture.read()
            if not ret:
                break
            if keep_flags[idx]:
                writer.write(frame)
                kept += 1
            elif deleted_writer is not None:
                deleted_writer.write(frame)
                deleted += 1
    finally:
        capture.release()
        writer.release()
        if deleted_writer is not None:
            deleted_writer.release()

    msg = (
        f"{video_path.name}: kept {kept}/{n_video} "
        f"({100 * kept / max(n_video, 1):.1f}%) -> {output_video.name}"
    )
    if write_deleted:
        msg += f"; nonspeaking {deleted} -> {output_deleted_video.name}"
    print(msg)
    return kept, deleted, n_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build speaking/nonspeaking videos from AU25_c on full clean OpenFace CSVs."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        type=Path,
        help="Face-crop mp4 paths (full OpenFace CSV must sit beside each video).",
    )
    parser.add_argument(
        "--keep-failed-tracks",
        action="store_true",
        help="Do not also require OpenFace success == 1.",
    )
    parser.add_argument(
        "--no-deleted",
        action="store_true",
        help="Do not write the complementary nonspeaking video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for video in args.videos:
        filter_video(
            video,
            require_success=not args.keep_failed_tracks,
            write_deleted=not args.no_deleted,
        )


if __name__ == "__main__":
    main()
