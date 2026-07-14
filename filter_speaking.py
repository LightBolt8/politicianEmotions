"""Build speaking/nonspeaking face-crop videos from full-video OpenFace AUs.

Speaking score:
  m_t = AU25_r + AU26_r
  Movement_t = SD(m_{t-w+1}, ..., m_t)   # ~2.4 s window at 5 fps

Default keep rule (within each candidate / debate video):
  τ* = Otsu threshold on Movement (successful frames)
  keep where Movement_t > otsu_factor * τ*

Optional modes: within-candidate z(Movement) > min_z, or fixed Movement cut.

Then consolidate into turns: bridge gaps ≤ max_gap_frames, drop turns
shorter than min_turn_frames.

Uses the full clean OpenFace CSV as a frame mask. Analysis OpenFace CSVs for
speaking clips should be produced by running OpenFace on *_speaking.mp4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

SPEAKING_AU_R = ("AU25_r", "AU26_r")

DEFAULT_WINDOW_FRAMES = 12
DEFAULT_MIN_Z = 0.0
DEFAULT_MODE = "otsu"
DEFAULT_OTSU_FACTOR = 0.85
# At 5 fps: gap=25 → ≤5 s pauses; min_turn=25 → drop <5 s blips.
DEFAULT_MAX_GAP_FRAMES = 25
DEFAULT_MIN_TURN_FRAMES = 25


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


def compute_m(df: pd.DataFrame) -> pd.Series:
    """m = AU25_r + AU26_r (mouth opening intensity)."""
    missing = [col for col in SPEAKING_AU_R if col not in df.columns]
    if missing:
        raise ValueError(f"Missing speaking AU columns: {', '.join(missing)}")
    au25 = pd.to_numeric(df["AU25_r"], errors="coerce").fillna(0.0)
    au26 = pd.to_numeric(df["AU26_r"], errors="coerce").fillna(0.0)
    return au25 + au26


def compute_movement(m: pd.Series, *, window_frames: int) -> pd.Series:
    """Movement_t = SD(m) over a rolling window."""
    if window_frames < 2:
        raise ValueError("window_frames must be >= 2")
    min_periods = max(2, window_frames // 2)
    return m.rolling(window_frames, min_periods=min_periods).std()


def zscore_within(
    series: pd.Series,
    *,
    ref_mask: pd.Series | None = None,
) -> pd.Series:
    """Z-score using mean/sd from ref_mask rows (default: all finite values)."""
    if ref_mask is None:
        ref = series.dropna()
    else:
        ref = series[ref_mask.fillna(False)].dropna()
    mu = float(ref.mean()) if len(ref) else 0.0
    sigma = float(ref.std(ddof=0)) if len(ref) else 0.0
    if not np.isfinite(sigma) or sigma < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def otsu_threshold(
    series: pd.Series,
    *,
    ref_mask: pd.Series | None = None,
) -> float:
    """Otsu threshold on Movement within one candidate/debate video."""
    if ref_mask is None:
        values = series.dropna().to_numpy(dtype=float)
    else:
        values = series[ref_mask.fillna(False)].dropna().to_numpy(dtype=float)
    if values.size < 2:
        return float("inf")
    if np.unique(values).size < 2:
        return float(values[0])
    return float(threshold_otsu(values))


def speaking_mask(
    df: pd.DataFrame,
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    mode: str = DEFAULT_MODE,
    min_z: float = DEFAULT_MIN_Z,
    min_movement: float | None = None,
    otsu_factor: float = DEFAULT_OTSU_FACTOR,
    ref_mask: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, float]:
    """
    Return (keep, m, movement, movement_z, threshold_used).

    mode:
      - otsu: Movement > otsu_factor * Otsu(τ*) within this video
      - z:    z(Movement) > min_z within this video
      - absolute: Movement > min_movement
    """
    m = compute_m(df)
    movement = compute_movement(m, window_frames=window_frames)
    movement_z = zscore_within(movement, ref_mask=ref_mask)

    mode = mode.lower()
    if mode == "otsu":
        if not (otsu_factor > 0):
            raise ValueError(f"otsu_factor must be > 0, got {otsu_factor}")
        tau = otsu_threshold(movement, ref_mask=ref_mask)
        threshold_used = float(tau * otsu_factor)
        keep = movement.fillna(0.0) > threshold_used
    elif mode == "z":
        keep = movement_z.fillna(0.0) > min_z
        threshold_used = float(min_z)
    elif mode == "absolute":
        if min_movement is None:
            raise ValueError("min_movement is required when mode=absolute")
        keep = movement.fillna(0.0) > min_movement
        threshold_used = float(min_movement)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (use otsu, z, or absolute)")

    return keep, m, movement, movement_z, threshold_used


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Return (start, end_exclusive, value) runs over a boolean array."""
    if mask.size == 0:
        return []
    runs: list[tuple[int, int, bool]] = []
    start = 0
    cur = bool(mask[0])
    for i in range(1, mask.size):
        v = bool(mask[i])
        if v != cur:
            runs.append((start, i, cur))
            start = i
            cur = v
    runs.append((start, mask.size, cur))
    return runs


def consolidate_turns(
    frame_keep: pd.Series,
    *,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    min_turn_frames: int = DEFAULT_MIN_TURN_FRAMES,
) -> tuple[pd.Series, list[tuple[int, int]]]:
    """
    Bridge nonspeaking gaps ≤ max_gap_frames; drop speaking islands
    shorter than min_turn_frames.
    """
    mask = frame_keep.fillna(False).to_numpy(dtype=bool).copy()

    if max_gap_frames > 0:
        for start, end, value in _runs(mask):
            if (not value) and (end - start) <= max_gap_frames:
                left_speak = start > 0 and mask[start - 1]
                right_speak = end < mask.size and mask[end]
                if left_speak and right_speak:
                    mask[start:end] = True

    if min_turn_frames > 0:
        for start, end, value in _runs(mask):
            if value and (end - start) < min_turn_frames:
                mask[start:end] = False

    turns = [(s, e) for s, e, v in _runs(mask) if v]
    return pd.Series(mask, index=frame_keep.index), turns


def filter_video(
    video_path: Path,
    csv_path: Path | None = None,
    output_video: Path | None = None,
    output_deleted_video: Path | None = None,
    output_turns_csv: Path | None = None,
    *,
    require_success: bool = True,
    write_deleted: bool = True,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    mode: str = DEFAULT_MODE,
    min_z: float = DEFAULT_MIN_Z,
    min_movement: float | None = None,
    otsu_factor: float = DEFAULT_OTSU_FACTOR,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    min_turn_frames: int = DEFAULT_MIN_TURN_FRAMES,
    use_turns: bool = True,
) -> tuple[int, int, int]:
    """
    Write speaking / nonspeaking videos + turns CSV.

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
    if output_turns_csv is None:
        output_turns_csv = video_path.with_name(f"{video_path.stem}_speaking_turns.csv")

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

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

    success_mask: pd.Series | None = None
    if require_success and "success" in df.columns:
        success_mask = pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1

    frame_keep, m, movement, movement_z, threshold_used = speaking_mask(
        df,
        window_frames=window_frames,
        mode=mode,
        min_z=min_z,
        min_movement=min_movement,
        otsu_factor=otsu_factor,
        ref_mask=success_mask,
    )
    if success_mask is not None:
        frame_keep = frame_keep & success_mask

    if use_turns:
        keep, turns = consolidate_turns(
            frame_keep,
            max_gap_frames=max_gap_frames,
            min_turn_frames=min_turn_frames,
        )
    else:
        keep = frame_keep
        turns = [(s, e) for s, e, v in _runs(keep.to_numpy(dtype=bool)) if v]

    turns_df = pd.DataFrame(
        [
            {
                "turn": i + 1,
                "start_frame": s + 1,
                "end_frame": e,
                "n_frames": e - s,
                "duration_sec": (e - s) / float(fps),
                "start_sec": s / float(fps),
                "end_sec": e / float(fps),
            }
            for i, (s, e) in enumerate(turns)
        ]
    )
    turns_df.to_csv(output_turns_csv, index=False)

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

    window_sec = window_frames / float(fps)
    if mode == "otsu":
        rule = f"Otsu τ*×{otsu_factor:g}={threshold_used:.3f}"
    elif mode == "z":
        rule = f"z(Movement)>{threshold_used:g}"
    else:
        rule = f"Movement>{threshold_used:g}"
    msg = (
        f"{video_path.name}: kept {kept}/{n_video} "
        f"({100 * kept / max(n_video, 1):.1f}%, {rule}, "
        f"window {window_frames}f≈{window_sec:.1f}s"
    )
    if use_turns:
        msg += (
            f", turns={len(turns)} "
            f"(gap≤{max_gap_frames}f min≥{min_turn_frames}f; "
            f"frame-level was {int(frame_keep.sum())})"
        )
    msg += f") -> {output_video.name}; turns -> {output_turns_csv.name}"
    if write_deleted:
        msg += f"; nonspeaking {deleted} -> {output_deleted_video.name}"
    print(msg)
    return kept, deleted, n_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build speaking/nonspeaking videos using Movement = rolling SD of "
            "m = AU25_r + AU26_r, with per-candidate Otsu×factor by default."
        )
    )
    parser.add_argument(
        "videos",
        nargs="+",
        type=Path,
        help="Face-crop mp4 paths (full OpenFace CSV must sit beside each video).",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=DEFAULT_WINDOW_FRAMES,
        help=f"Rolling SD window in frames (default: {DEFAULT_WINDOW_FRAMES}).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Optional window length in seconds (overrides --window-frames).",
    )
    parser.add_argument(
        "--mode",
        choices=("otsu", "z", "absolute"),
        default=DEFAULT_MODE,
        help="Threshold mode (default: otsu). z uses --min-z; absolute uses --min-movement.",
    )
    parser.add_argument(
        "--min-z",
        type=float,
        default=DEFAULT_MIN_Z,
        help="For --mode z: keep when within-candidate z(Movement) > this (default: 0).",
    )
    parser.add_argument(
        "--min-movement",
        type=float,
        default=None,
        help="For --mode absolute: keep when Movement > this.",
    )
    parser.add_argument(
        "--otsu-factor",
        type=float,
        default=DEFAULT_OTSU_FACTOR,
        help=f"For --mode otsu: keep when Movement > factor * τ* (default: {DEFAULT_OTSU_FACTOR}).",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=DEFAULT_MAX_GAP_FRAMES,
        help=f"Bridge gaps ≤ this many frames (default: {DEFAULT_MAX_GAP_FRAMES}).",
    )
    parser.add_argument(
        "--min-turn-frames",
        type=int,
        default=DEFAULT_MIN_TURN_FRAMES,
        help=f"Drop speaking islands shorter than this (default: {DEFAULT_MIN_TURN_FRAMES}).",
    )
    parser.add_argument(
        "--no-turns",
        action="store_true",
        help="Keep raw frame-level mask (skip turn consolidation).",
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
        window_frames = args.window_frames
        if args.window_seconds is not None:
            capture = cv2.VideoCapture(str(video.expanduser().resolve()))
            fps = capture.get(cv2.CAP_PROP_FPS) or 5.0
            capture.release()
            window_frames = max(2, int(round(args.window_seconds * fps)))
        filter_video(
            video,
            require_success=not args.keep_failed_tracks,
            write_deleted=not args.no_deleted,
            window_frames=window_frames,
            mode=args.mode,
            min_z=args.min_z,
            min_movement=args.min_movement,
            otsu_factor=args.otsu_factor,
            max_gap_frames=args.max_gap_frames,
            min_turn_frames=args.min_turn_frames,
            use_turns=not args.no_turns,
        )


if __name__ == "__main__":
    main()
