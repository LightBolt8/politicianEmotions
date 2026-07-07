"""Extract and crop candidate faces from a debate video using ArcFace."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis


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


def calibrate_from_video(
    app: FaceAnalysis,
    video_path: Path,
    photo_embeddings: list[np.ndarray],
    calibrate_seconds: float,
) -> list[np.ndarray]:
    """
    Build reference embeddings from a split-screen debate frame so both
    candidates score ~0.80+ against their own footage.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    for offset in (0, 30, 60, 120, -60, -120):
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0, (calibrate_seconds + offset) * 1000))
        ret, frame = capture.read()
        if not ret:
            continue

        faces = app.get(frame)
        if len(faces) < 2:
            continue

        debate_embeddings: list[np.ndarray | None] = [None, None]
        for face in faces:
            photo_sims = [
                cosine_similarity(face.embedding, photo) for photo in photo_embeddings
            ]
            candidate_idx = int(np.argmax(photo_sims))
            if debate_embeddings[candidate_idx] is None:
                debate_embeddings[candidate_idx] = face.embedding

        if all(embedding is not None for embedding in debate_embeddings):
            capture.release()
            print(f"Calibrated references from {calibrate_seconds + offset:.0f}s in video")
            return debate_embeddings  # type: ignore[return-value]

    capture.release()
    raise RuntimeError(
        "Could not calibrate from video: no split-screen frame with two faces found. "
        "Try a different --calibrate-seconds value."
    )


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


def process_video(
    app: FaceAnalysis,
    input_video_path: Path,
    output_a_path: Path,
    output_b_path: Path,
    known_embeddings: list[np.ndarray],
    *,
    fps: int = 5,
    frame_size: tuple[int, int] = (256, 256),
    skip_rate: int = 6,
    similarity_threshold: float = 0.7,
    max_yaw_deg: float = 30.0,
    start_seconds: float = 0.0,
    max_seconds: float | None = None,
    progress_interval: int = 600,
) -> tuple[int, int]:
    """Detect faces with ArcFace, match candidates, and write cropped-face videos."""
    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer_a = cv2.VideoWriter(str(output_a_path), fourcc, fps, frame_size)
    writer_b = cv2.VideoWriter(str(output_b_path), fourcc, fps, frame_size)
    if not writer_a.isOpened() or not writer_b.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_a = cv2.VideoWriter(str(output_a_path), fourcc, fps, frame_size)
        writer_b = cv2.VideoWriter(str(output_b_path), fourcc, fps, frame_size)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = None
    if max_seconds is not None:
        max_frame = int((start_seconds + max_seconds) * source_fps)

    frame_count = int(start_seconds * source_fps)
    written_a = 0
    written_b = 0

    try:
        while True:
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
                if not is_frontal(face, max_yaw_deg):
                    continue

                candidate_idx = match_candidate(
                    face.embedding, known_embeddings, similarity_threshold
                )
                if candidate_idx is None:
                    continue

                cropped_face = crop_face_with_padding(frame, face.bbox)
                if cropped_face is None:
                    continue

                resized_face = cv2.resize(cropped_face, frame_size)

                if candidate_idx == 0:
                    writer_a.write(resized_face)
                    written_a += 1
                else:
                    writer_b.write(resized_face)
                    written_b += 1
    finally:
        capture.release()
        writer_a.release()
        writer_b.release()

    return written_a, written_b


def resolve_output_paths(
    input_video: Path,
    output_dir: Path,
    output_a: Path | None,
    output_b: Path | None,
) -> tuple[Path, Path]:
    """Place exports in output_dir/<debate>/<candidate>/ unless paths are given explicitly."""
    run_dir = output_dir / input_video.stem
    path_a = output_a or run_dir / "candidate_A_clean" / "candidate_A_clean.mp4"
    path_b = output_b or run_dir / "candidate_B_clean" / "candidate_B_clean.mp4"
    if not output_a:
        path_a.parent.mkdir(parents=True, exist_ok=True)
    if not output_b:
        path_b.parent.mkdir(parents=True, exist_ok=True)
    return path_a, path_b


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
        help="Reference photo for candidate A.",
    )
    parser.add_argument(
        "--candidate-b",
        type=Path,
        default=Path("candidate_B.jpg"),
        help="Reference photo for candidate B.",
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
        default=400.0,
        help="Skip this many seconds at the start (default skips intro).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional limit on how many seconds of video to process.",
    )
    parser.add_argument(
        "--calibrate-seconds",
        type=float,
        default=500.0,
        help="Extract reference faces from this timestamp in the debate video.",
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
        default=0.7,
        help="Minimum cosine similarity to match a candidate (0-1, higher = stricter).",
    )
    parser.add_argument(
        "--max-yaw",
        type=float,
        default=30.0,
        help="Skip faces turned more than this many degrees left/right (yaw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_video = args.input_video.expanduser()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not args.candidate_a.exists():
        raise FileNotFoundError(f"Reference image not found: {args.candidate_a}")
    if not args.candidate_b.exists():
        raise FileNotFoundError(f"Reference image not found: {args.candidate_b}")

    print("Loading ArcFace model...")
    app = create_face_app()
    photo_embeddings = build_known_embeddings(app, args.candidate_a, args.candidate_b)
    known_embeddings = calibrate_from_video(
        app, input_video, photo_embeddings, args.calibrate_seconds
    )

    for label, embedding in zip(("A", "B"), known_embeddings):
        sim = cosine_similarity(embedding, embedding)
        print(f"  Candidate {label} self-similarity: {sim:.3f}")

    output_a, output_b = resolve_output_paths(
        input_video, args.output_dir, args.output_a, args.output_b
    )
    written_a, written_b = process_video(
        app,
        input_video,
        output_a,
        output_b,
        known_embeddings,
        fps=args.fps,
        frame_size=(args.frame_size, args.frame_size),
        skip_rate=args.skip_rate,
        similarity_threshold=args.threshold,
        max_yaw_deg=args.max_yaw,
        start_seconds=args.start_seconds,
        max_seconds=args.max_seconds,
    )
    print(f"Wrote {written_a} frames to {output_a}")
    print(f"Wrote {written_b} frames to {output_b}")
    if written_a == 0 and written_b == 0:
        raise RuntimeError(
            "No faces matched either candidate. Check reference photos and "
            "try adjusting --start-seconds or --threshold."
        )


if __name__ == "__main__":
    main()
