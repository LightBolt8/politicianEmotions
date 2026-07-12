"""Extract and crop candidate faces from a debate video using ArcFace."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Set by SIGINT/SIGTERM so the main loop can finalize writers cleanly.
_stop_requested = False


def _request_stop(signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True
    print(f"\nReceived signal {signum}; finishing current frame and finalizing outputs...")


def create_face_app() -> FaceAnalysis:
    """Load ArcFace model with face detection (CPU)."""
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_reference_embedding(app: FaceAnalysis, image_path: Path) -> np.ndarray:
    """Load an image and return its 512-dimensional ArcFace embedding."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    faces = app.get(image)
    if not faces:
        raise ValueError(f"No face found in reference image: {image_path}")
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).embedding


def build_known_embeddings(
    app: FaceAnalysis,
    candidate_a_path: Path,
    candidate_b_path: Path,
) -> list[np.ndarray]:
    """Build ArcFace reference embeddings for both candidates from photos."""
    return [
        load_reference_embedding(app, candidate_a_path),
        load_reference_embedding(app, candidate_b_path),
    ]


def match_candidate(
    embedding: np.ndarray,
    known_embeddings: list[np.ndarray],
    threshold: float,
) -> int | None:
    """Return the best-matching candidate if similarity >= threshold."""
    similarities = [cosine_similarity(embedding, known) for known in known_embeddings]
    best_idx = int(np.argmax(similarities))
    if similarities[best_idx] >= threshold:
        return best_idx
    return None


def is_frontal(face: insightface.app.common.Face, max_yaw_deg: float) -> bool:
    """Return True if face yaw is within +/- max_yaw_deg of frontal."""
    pose = getattr(face, "pose", None)
    if pose is None:
        return False
    return abs(float(pose[0])) <= max_yaw_deg


def crop_face_with_padding(
    frame: np.ndarray,
    bbox: np.ndarray,
    padding_ratio: float = 0.4,
) -> np.ndarray | None:
    """Crop a face region from a frame with padding around the bounding box."""
    x1, y1, x2, y2 = bbox.astype(int)
    height, width = frame.shape[:2]
    pad = int((y2 - y1) * padding_ratio)
    y1, y2 = max(0, y1 - pad), min(height, y2 + pad)
    x1, x2 = max(0, x1 - pad), min(width, x2 + pad)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    return cropped


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


def part_path(final_path: Path, part_index: int) -> Path:
    return final_path.with_name(f"{final_path.stem}.part{part_index:03d}{final_path.suffix}")


def checkpoint_path_for(deleted_path: Path) -> Path:
    return deleted_path.parent / "preprocess_checkpoint.json"


def is_playable_video(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    capture = cv2.VideoCapture(str(path))
    ok = capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    if ok:
        ret, _ = capture.read()
        ok = bool(ret)
    capture.release()
    return ok


def list_part_files(final_path: Path) -> list[Path]:
    pattern = f"{final_path.stem}.part*{final_path.suffix}"
    return sorted(final_path.parent.glob(pattern))


def concat_parts(final_path: Path, fps: float, frame_size: tuple[int, int]) -> int:
    """Concatenate finalized part files into final_path. Returns total frames written."""
    parts = [p for p in list_part_files(final_path) if is_playable_video(p)]
    if not parts:
        return 0

    if len(parts) == 1:
        parts[0].replace(final_path)
        return int(cv2.VideoCapture(str(final_path)).get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    list_file = final_path.with_suffix(".concat.txt")
    list_file.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in parts),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(final_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and is_playable_video(final_path):
            for part in parts:
                part.unlink(missing_ok=True)
            return int(cv2.VideoCapture(str(final_path)).get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        list_file.unlink(missing_ok=True)

    # Fallback: re-encode with OpenCV if ffmpeg copy fails.
    writer = open_video_writer(final_path, fps, frame_size)
    total = 0
    try:
        for part in parts:
            capture = cv2.VideoCapture(str(part))
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
                    frame = cv2.resize(frame, frame_size)
                writer.write(frame)
                total += 1
            capture.release()
    finally:
        writer.release()

    for part in parts:
        part.unlink(missing_ok=True)
    return total


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def process_video(
    app: FaceAnalysis,
    input_video_path: Path,
    output_a_path: Path,
    output_b_path: Path,
    deleted_path: Path,
    known_embeddings: list[np.ndarray],
    *,
    fps: int = 5,
    frame_size: tuple[int, int] = (256, 256),
    skip_rate: int = 6,
    similarity_threshold: float = 0.6,
    max_yaw_deg: float = 30.0,
    start_seconds: float = 0.0,
    max_seconds: float | None = None,
    progress_interval: int = 600,
    resume: bool = False,
    checkpoint_every_frames: int = 9000,
) -> tuple[int, int, int, bool]:
    """
    Detect faces, write matched/rejected crops as part files, and checkpoint progress.

    Returns (written_a, written_b, written_deleted, completed).
    completed is False if stopped early via signal.
    """
    global _stop_requested
    _stop_requested = False

    checkpoint_path = checkpoint_path_for(deleted_path)
    source_fps_probe = cv2.VideoCapture(str(input_video_path))
    if not source_fps_probe.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    source_fps = source_fps_probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(source_fps_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps_probe.release()

    written_a = 0
    written_b = 0
    written_deleted = 0
    part_index = 0
    frame_count = int(start_seconds * source_fps)

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt.get("complete"):
            print(f"Checkpoint already complete: {checkpoint_path}")
            return (
                int(ckpt.get("written_a", 0)),
                int(ckpt.get("written_b", 0)),
                int(ckpt.get("written_deleted", 0)),
                True,
            )
        frame_count = int(ckpt["last_frame"])
        written_a = int(ckpt.get("written_a", 0))
        written_b = int(ckpt.get("written_b", 0))
        written_deleted = int(ckpt.get("written_deleted", 0))
        part_index = int(ckpt.get("part_index", 0))
        # Drop a corrupt in-progress part left by a hard kill.
        for final in (output_a_path, output_b_path, deleted_path):
            candidate = part_path(final, part_index)
            if candidate.exists() and not is_playable_video(candidate):
                print(f"Removing incomplete part {candidate}")
                candidate.unlink(missing_ok=True)
        print(
            f"Resuming from frame {frame_count} "
            f"(part {part_index:03d}, kept A/B/deleted={written_a}/{written_b}/{written_deleted})"
        )

    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    # Seek to resume/start position.
    if frame_count > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    max_frame = None
    if max_seconds is not None:
        max_frame = int((start_seconds + max_seconds) * source_fps)

    writer_a = open_video_writer(part_path(output_a_path, part_index), fps, frame_size)
    writer_b = open_video_writer(part_path(output_b_path, part_index), fps, frame_size)
    writer_deleted = open_video_writer(part_path(deleted_path, part_index), fps, frame_size)
    frames_in_part = 0
    completed = True

    def release_writers() -> None:
        writer_a.release()
        writer_b.release()
        writer_deleted.release()

    def write_checkpoint(next_part: int) -> None:
        save_checkpoint(
            checkpoint_path,
            {
                "input_video": str(input_video_path),
                "output_a": str(output_a_path),
                "output_b": str(output_b_path),
                "deleted": str(deleted_path),
                "last_frame": frame_count,
                "written_a": written_a,
                "written_b": written_b,
                "written_deleted": written_deleted,
                "part_index": next_part,
                "fps": fps,
                "frame_size": list(frame_size),
                "skip_rate": skip_rate,
                "complete": False,
            },
        )

    def rotate_part() -> None:
        nonlocal writer_a, writer_b, writer_deleted, part_index, frames_in_part
        release_writers()
        # Remove empty parts so concat stays clean.
        for final in (output_a_path, output_b_path, deleted_path):
            path = part_path(final, part_index)
            if path.exists() and not is_playable_video(path):
                path.unlink(missing_ok=True)
        part_index += 1
        frames_in_part = 0
        write_checkpoint(part_index)
        print(f"Checkpointed at frame {frame_count} -> part {part_index:03d}")
        writer_a = open_video_writer(part_path(output_a_path, part_index), fps, frame_size)
        writer_b = open_video_writer(part_path(output_b_path, part_index), fps, frame_size)
        writer_deleted = open_video_writer(
            part_path(deleted_path, part_index), fps, frame_size
        )

    not_frontal = 0
    no_match = 0
    crop_failed = 0

    try:
        while True:
            if _stop_requested:
                completed = False
                break

            ret, frame = capture.read()
            if not ret:
                break

            frame_count += 1
            if max_frame is not None and frame_count > max_frame:
                break

            if progress_interval and frame_count % progress_interval == 0:
                end = max_frame or total_frames
                print(f"Processing frame {frame_count} / {end or '?'}")

            if frame_count % skip_rate != 0:
                continue

            for face in app.get(frame):
                cropped_face = crop_face_with_padding(frame, face.bbox)
                if cropped_face is None:
                    continue
                resized_face = cv2.resize(cropped_face, frame_size)

                if not is_frontal(face, max_yaw_deg):
                    writer_deleted.write(resized_face)
                    written_deleted += 1
                    frames_in_part += 1
                    continue

                candidate_idx = match_candidate(
                    face.embedding, known_embeddings, similarity_threshold
                )
                if candidate_idx is None:
                    writer_deleted.write(resized_face)
                    written_deleted += 1
                    frames_in_part += 1
                    continue

                if candidate_idx == 0:
                    writer_a.write(resized_face)
                    written_a += 1
                else:
                    writer_b.write(resized_face)
                    written_b += 1
                frames_in_part += 1

            if (
                checkpoint_every_frames > 0
                and frame_count % checkpoint_every_frames == 0
                and frames_in_part > 0
            ):
                rotate_part()
    finally:
        capture.release()
        release_writers()
        # Drop empty trailing part.
        for final in (output_a_path, output_b_path, deleted_path):
            path = part_path(final, part_index)
            if path.exists() and not is_playable_video(path):
                path.unlink(missing_ok=True)

        if completed:
            print("Assembling final videos from parts...")
            concat_parts(output_a_path, fps, frame_size)
            concat_parts(output_b_path, fps, frame_size)
            concat_parts(deleted_path, fps, frame_size)
            save_checkpoint(
                checkpoint_path,
                {
                    "input_video": str(input_video_path),
                    "output_a": str(output_a_path),
                    "output_b": str(output_b_path),
                    "deleted": str(deleted_path),
                    "last_frame": frame_count,
                    "written_a": written_a,
                    "written_b": written_b,
                    "written_deleted": written_deleted,
                    "part_index": part_index,
                    "fps": fps,
                    "frame_size": list(frame_size),
                    "skip_rate": skip_rate,
                    "complete": True,
                },
            )
        else:
            # Next resume opens a fresh part after the ones we just finalized.
            write_checkpoint(part_index + 1)
            print(
                f"Stopped early at frame {frame_count}. "
                f"Resume with --resume (checkpoint: {checkpoint_path})"
            )

    return written_a, written_b, written_deleted, completed


def resolve_output_paths(
    input_video: Path,
    output_dir: Path,
    output_a: Path | None,
    output_b: Path | None,
) -> tuple[Path, Path, Path]:
    """Place exports in output_dir/<debate>/<candidate>/ unless paths are given explicitly."""
    run_dir = output_dir / input_video.stem
    path_a = output_a or run_dir / "candidate_A_clean" / "candidate_A_clean.mp4"
    path_b = output_b or run_dir / "candidate_B_clean" / "candidate_B_clean.mp4"
    deleted = run_dir / "deleted_faces.mp4"
    if output_a is not None:
        deleted = output_a.parent.parent / "deleted_faces.mp4"
    path_a.parent.mkdir(parents=True, exist_ok=True)
    path_b.parent.mkdir(parents=True, exist_ok=True)
    deleted.parent.mkdir(parents=True, exist_ok=True)
    return path_a, path_b, deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract candidate face crops from a debate video using ArcFace."
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        required=True,
        help="Path to the source debate video.",
    )
    parser.add_argument(
        "--candidate-a",
        type=Path,
        default=Path("candidate_A.jpg"),
        help="Reference photo for candidate A (used as-is for matching).",
    )
    parser.add_argument(
        "--candidate-b",
        type=Path,
        default=Path("candidate_B.jpg"),
        help="Reference photo for candidate B (used as-is for matching).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Exported"),
        help="Directory where exported face videos are saved.",
    )
    parser.add_argument(
        "--output-a",
        type=Path,
        default=None,
        help="Output video for candidate A (default: Exported/<debate>/candidate_A_clean/candidate_A_clean.mp4).",
    )
    parser.add_argument(
        "--output-b",
        type=Path,
        default=None,
        help="Output video for candidate B (default: Exported/<debate>/candidate_B_clean/candidate_B_clean.mp4).",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Skip this many seconds at the start of the main export pass (default: 0).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional limit on how many seconds of video to process.",
    )
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument(
        "--frame-size",
        type=int,
        default=256,
        help="Output face crop resolution in pixels (square, e.g. 512 for higher quality).",
    )
    parser.add_argument("--skip-rate", type=int, default=6)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Minimum cosine similarity to match a candidate (0-1, higher = stricter).",
    )
    parser.add_argument(
        "--max-yaw",
        type=float,
        default=30.0,
        help="Skip faces turned more than this many degrees left/right (yaw).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from Exported/<debate>/preprocess_checkpoint.json and append new parts.",
    )
    parser.add_argument(
        "--checkpoint-every-frames",
        type=int,
        default=9000,
        help="Finalize a part and write checkpoint every N source frames (default: 9000 ≈5 min at 30fps).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    input_video = args.input_video.expanduser()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not args.candidate_a.exists():
        raise FileNotFoundError(f"Reference image not found: {args.candidate_a}")
    if not args.candidate_b.exists():
        raise FileNotFoundError(f"Reference image not found: {args.candidate_b}")

    print("Loading ArcFace model...")
    app = create_face_app()
    print("Loading embeddings from reference images...")
    known_embeddings = build_known_embeddings(app, args.candidate_a, args.candidate_b)

    for label, embedding in zip(("A", "B"), known_embeddings):
        sim = cosine_similarity(embedding, embedding)
        print(f"  Candidate {label} self-similarity: {sim:.3f}")

    output_a, output_b, deleted_path = resolve_output_paths(
        input_video, args.output_dir, args.output_a, args.output_b
    )

    if not args.resume:
        # Fresh run: clear old parts/finals/checkpoint for these outputs.
        ckpt = checkpoint_path_for(deleted_path)
        ckpt.unlink(missing_ok=True)
        for final in (output_a, output_b, deleted_path):
            final.unlink(missing_ok=True)
            for part in list_part_files(final):
                part.unlink(missing_ok=True)

    written_a, written_b, written_deleted, completed = process_video(
        app,
        input_video,
        output_a,
        output_b,
        deleted_path,
        known_embeddings,
        fps=args.fps,
        frame_size=(args.frame_size, args.frame_size),
        skip_rate=args.skip_rate,
        similarity_threshold=args.threshold,
        max_yaw_deg=args.max_yaw,
        start_seconds=args.start_seconds,
        max_seconds=args.max_seconds,
        resume=args.resume,
        checkpoint_every_frames=args.checkpoint_every_frames,
    )
    print(f"Wrote {written_a} frames to {output_a}")
    print(f"Wrote {written_b} frames to {output_b}")
    print(f"Wrote {written_deleted} rejected faces to {deleted_path}")
    if not completed:
        print("Run incomplete — re-run with the same args plus --resume to continue.")
        sys.exit(2)
    if written_a == 0 and written_b == 0:
        raise RuntimeError(
            "No faces matched either candidate. Check reference photos and "
            "try adjusting --start-seconds or --threshold."
        )


if __name__ == "__main__":
    main()
