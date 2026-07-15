"""Validation export: 3 one-minute Dataset clips × ~5 fps (every 6th frame).

For each sampled Dataset frame:
  1. Accurately cut 1-min clips (ffmpeg -ss after -i), read them sequentially
     (avoid OpenCV seeks on the full Dataset, which can desync timestamps).
  2. ArcFace match → matched 0/1
  3. Face crop → per-window per-candidate clip → OpenFace
     (windows are not concatenated — Movement / turns cannot leak across
     15–16 / 45–46 / 75–76).
  4. Speaking = Movement > τ_mov AND m > τ_m (thresholds from full clean
     CSV; dual-gate Otsu×factors; no new Otsu on shorts), then
     consolidate_turns. Same per-candidate rules as filter_speaking.

Output: newExported/time_windows/{year}_new.csv — one row per sampled frame
with both candidates side-by-side ({Name}_matched, {Name}_speaking).
Rows kept if at least one candidate matched.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

from filter_speaking import (
    DEFAULT_MAX_GAP_FRAMES,
    DEFAULT_MIN_TURN_FRAMES,
    DEFAULT_WINDOW_FRAMES,
    compute_m,
    consolidate_turns,
    otsu_threshold,
    speaking_mask,
)
from preprocessing_data import (
    build_known_embeddings,
    is_frontal,
    match_candidate,
)
from run_openface import find_feature_extraction, run_openface_on_video

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "newExported" / "time_windows"
CLIPS = OUT / "clips"
WORK = OUT / "work"
DATA = ROOT / "newExported"

SKIP = 6  # ~5 fps — same as preprocess; rolling SD window ≈ 2.4s
THRESHOLD = 0.6
MAX_YAW = 30.0
OTSU_FACTOR = 0.75  # Movement Otsu factor (dual gate with M_OTSU_FACTOR)
M_OTSU_FACTOR = 0.85  # m = AU25_r+AU26_r Otsu factor
FRAME_SIZE = (256, 256)
CROP_FPS = 5.0  # crop strip rate matches SKIP sampling on 30fps source

WINDOWS = (
    ("15-16min", 15 * 60.0, 16 * 60.0),
    ("45-46min", 45 * 60.0, 46 * 60.0),
    ("75-76min", 75 * 60.0, 76 * 60.0),
)

YEARS = (
    (
        "2004",
        "2004",
        ("Bush", "Kerry"),
        (
            DATA / "2004/Bush_clean_2004/Bush_clean_2004.csv",
            DATA / "2004/Kerry_clean_2004/Kerry_clean_2004.csv",
        ),
        ROOT / "refs/2004/candidate_A.jpg",
        ROOT / "refs/2004/candidate_B.jpg",
    ),
    (
        "2008",
        "2008",
        ("McCain", "Obama"),
        (
            DATA / "2008/McCain_clean_2008/McCain_clean_2008.csv",
            DATA / "2008/Obama_clean_2008/Obama_clean_2008.csv",
        ),
        ROOT / "refs/2008/candidate_A.jpg",
        ROOT / "refs/2008/candidate_B.jpg",
    ),
    (
        "2012",
        "2012",
        ("Obama", "Romney"),
        (
            DATA / "2012/Obama_clean_2012/Obama_clean_2012.csv",
            DATA / "2012/Romney_clean_2012/Romney_clean_2012.csv",
        ),
        ROOT / "refs/2012/candidate_A.jpg",
        ROOT / "refs/2012/candidate_B.jpg",
    ),
    (
        "2016",
        "2016",
        ("Trump", "Clinton"),
        (
            DATA / "2016/Trump_clean_2016/Trump_clean_2016.csv",
            DATA / "2016/Clinton_clean_2016/Clinton_clean_2016.csv",
        ),
        ROOT / "refs/2016/candidate_A.jpg",
        ROOT / "refs/2016/candidate_B.jpg",
    ),
    (
        "2020",
        "2020",
        ("Trump", "Biden"),
        (
            DATA / "2020/Trump_clean_2020/Trump_clean_2020.csv",
            DATA / "2020/Biden_clean_2020/Biden_clean_2020.csv",
        ),
        ROOT / "refs/2020/candidate_A.jpg",
        ROOT / "refs/2020/candidate_B.jpg",
    ),
    (
        "2024b",
        "2024b",
        ("Trump", "Biden"),
        (
            DATA / "2024b/Trump_clean_2024b/Trump_clean_2024b.csv",
            DATA / "2024b/Biden_clean_2024b/Biden_clean_2024b.csv",
        ),
        ROOT / "refs/2024b/candidate_A.jpg",
        ROOT / "refs/2024b/candidate_B.jpg",
    ),
    (
        "2024k",
        "2024k",
        ("Trump", "Harris"),
        (
            DATA / "2024k/Trump_clean_2024/Trump_clean_2024.csv",
            DATA / "2024k/Harris_clean_2024/Harris_clean_2024.csv",
        ),
        ROOT / "refs/2024k/candidate_A.jpg",
        ROOT / "refs/2024k/candidate_B.jpg",
    ),
)


def mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m}:{s:06.3f}"


def source_props(path: Path) -> tuple[int, float, float]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return n, fps, (n / fps if fps > 0 else 0.0)


def ffmpeg_cut(
    dataset: Path,
    start: float,
    end: float,
    dst: Path,
    *,
    force: bool = False,
) -> None:
    """Cut [start, end) with accurate timestamps (-ss after -i)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 1000 and not force:
        print(f"skip existing {dst.name}")
        return
    # -ss after -i is slower but frame-accurate (needed so clip time == debate time).
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(dataset),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{end - start:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-avoid_negative_ts",
        "make_zero",
        str(dst),
    ]
    print(f"cut {dataset.name} [{start:.0f}-{end:.0f}s] -> {dst.name}", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)


def create_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def crop_face(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    from preprocessing_data import crop_face_with_padding

    cropped = crop_face_with_padding(frame, bbox)
    if cropped is None:
        return None
    return cv2.resize(cropped, FRAME_SIZE)


def taus_from_full_clean(clean_csv: Path) -> tuple[float, float]:
    """Reuse full-debate dual-gate thresholds; do not refit Otsu on short clips."""
    df = pd.read_csv(clean_csv)
    df.columns = [c.strip() for c in df.columns]
    success = None
    if "success" in df.columns:
        success = pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1
    *_rest, tau_mov = speaking_mask(
        df, otsu_factor=OTSU_FACTOR, ref_mask=success
    )
    m = compute_m(df)
    tau_m = float(otsu_threshold(m, ref_mask=success) * M_OTSU_FACTOR)
    return float(tau_mov), tau_m


def write_crop_video(crops: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, CROP_FPS, FRAME_SIZE)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, CROP_FPS, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {path}")
    for crop in crops:
        writer.write(crop)
    writer.release()


def speaking_on_crops(
    crops: list[np.ndarray],
    tau_mov: float,
    tau_m: float,
    work_dir: Path,
    *,
    openface_bin: Path,
    openface_cwd: Path | None,
    warmup_frames: int = DEFAULT_WINDOW_FRAMES,
) -> list[int]:
    """OpenFace one window's crops; Movement > τ_mov AND m > τ_m, then turns.

    Prepends warmup_frames copies of the first crop so the rolling Movement
    window is defined at clip t=0; pad flags are discarded. Call once per
    window so Movement/turns never span 15–16 → 45–46 → 75–76.
    """
    if not crops:
        return []
    pad_n = max(0, int(warmup_frames))
    padded = ([crops[0]] * pad_n + list(crops)) if pad_n else list(crops)
    video = work_dir / "crops.mp4"
    write_crop_video(padded, video)
    run_openface_on_video(openface_bin, video, work_dir, openface_cwd=openface_cwd)
    csv_path = work_dir / f"{video.stem}.csv"
    if not csv_path.is_file():
        candidates = list(work_dir.glob("*.csv"))
        if not candidates:
            return [0] * len(crops)
        csv_path = candidates[0]

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    keep, m, _movement, _mz, _thr = speaking_mask(
        df,
        window_frames=DEFAULT_WINDOW_FRAMES,
        mode="absolute",
        min_movement=tau_mov,
        otsu_factor=OTSU_FACTOR,
    )
    keep = keep & (m.fillna(0.0) > float(tau_m))
    keep, _turns = consolidate_turns(
        keep.fillna(False),
        max_gap_frames=DEFAULT_MAX_GAP_FRAMES,
        min_turn_frames=DEFAULT_MIN_TURN_FRAMES,
    )
    flags = keep.astype(int).tolist()
    flags = flags[pad_n : pad_n + len(crops)]
    if len(flags) < len(crops):
        flags.extend([0] * (len(crops) - len(flags)))
    return flags[: len(crops)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs="+", metavar="YEAR", help="Subset of years.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    selected = list(YEARS)
    if args.years:
        want = set(args.years)
        selected = [y for y in YEARS if y[0] in want]
        missing = want - {y[0] for y in selected}
        if missing:
            raise SystemExit(f"Unknown year(s): {sorted(missing)}")

    print("Loading ArcFace...", flush=True)
    app = create_face_app()
    openface_bin = find_feature_extraction(None)
    openface_cwd = (
        openface_bin.parent.parent.parent
        if openface_bin.parent.name == "bin"
        else None
    )
    if openface_cwd is not None and not (openface_cwd / "model").is_dir():
        openface_cwd = None

    for year, stem, names, cleans, ref_a, ref_b in selected:
        dataset = ROOT / "Dataset" / f"{stem}.mp4"
        if not dataset.is_file() or not all(c.is_file() for c in cleans):
            print(f"SKIP {year}: missing dataset or clean CSV")
            continue

        _, fps, src_dur = source_props(dataset)
        print(f"\n=== {year} ({src_dur/60:.1f} min) ===", flush=True)

        embeddings = build_known_embeddings(app, ref_a, ref_b)
        taus = {
            name: taus_from_full_clean(clean)
            for name, clean in zip(names, cleans, strict=True)
        }
        for name, (tau_mov, tau_m) in taus.items():
            print(
                f"  {name}: mov×{OTSU_FACTOR:g}={tau_mov:.4f} "
                f"AND m×{M_OTSU_FACTOR:g}={tau_m:.4f}",
                flush=True,
            )

        # Per candidate: list of (row_dict_without_speaking, crop_or_None)
        pending: dict[str, list[tuple[dict, np.ndarray | None]]] = {
            n: [] for n in names
        }

        # Read accurately cut 1-min clips sequentially (OpenCV Dataset seeks are wrong).
        for window, t0, t1 in WINDOWS:
            if t0 >= src_dur:
                continue
            end = min(t1, src_dur)
            clip = CLIPS / f"source_{year}_{window}.mp4"
            ffmpeg_cut(dataset, t0, end, clip, force=False)

            clip_cap = cv2.VideoCapture(str(clip))
            if not clip_cap.isOpened():
                raise RuntimeError(f"Cannot open clip {clip}")
            clip_fps = float(clip_cap.get(cv2.CAP_PROP_FPS) or fps)
            idx = 0
            while True:
                ok, frame = clip_cap.read()
                if not ok or frame is None:
                    break
                clip_ts = idx / clip_fps
                ts = t0 + clip_ts
                if ts >= end - 1e-9:
                    break
                # Same cadence as preprocess (~every SKIP frames at source fps).
                if idx % SKIP != 0:
                    idx += 1
                    continue
                src_frame = int(round(ts * fps))

                base_row = {
                    "year": year,
                    "window": window,
                    "clip_file": clip.name,
                    "frame": src_frame,
                    "timestamp": round(ts, 6),
                    "timestamp_mmss": mmss(ts),
                    "clip_timestamp": round(clip_ts, 6),
                }
                crops_here: dict[str, np.ndarray] = {}
                for face in app.get(frame):
                    if not is_frontal(face, MAX_YAW):
                        continue
                    cand = match_candidate(face.embedding, embeddings, THRESHOLD)
                    if cand is None:
                        continue
                    crop = crop_face(frame, face.bbox)
                    if crop is None:
                        continue
                    crops_here[names[cand]] = crop

                for name in names:
                    crop = crops_here.get(name)
                    pending[name].append(
                        (
                            {
                                **base_row,
                                "candidate": name,
                                "matched": 1 if crop is not None else 0,
                            },
                            crop,
                        )
                    )
                idx += 1
            clip_cap.release()

        # OpenFace + speaking per window (absolute τ* from full clean).
        # Do not concatenate windows — that leaked Movement across clip boundaries.
        speaking_by_name: dict[str, list[int]] = {}
        matched_by_name: dict[str, list[int]] = {}
        base_rows: list[dict] | None = None

        for name in names:
            items = pending[name]
            if base_rows is None:
                base_rows = [
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("candidate", "matched")
                    }
                    for row, _ in items
                ]
            speaking = [0] * len(items)
            matched_by_name[name] = [int(row["matched"]) for row, _ in items]

            by_window: dict[str, list[int]] = {}
            for i, (row, crop) in enumerate(items):
                if crop is None:
                    continue
                by_window.setdefault(row["window"], []).append(i)

            total = sum(len(v) for v in by_window.values())
            print(
                f"  {name}: {total} matched crops → OpenFace per window "
                f"(τ_mov={taus[name][0]:.4f}, τ_m={taus[name][1]:.4f})",
                flush=True,
            )
            for window, crop_indices in by_window.items():
                crops = [items[i][1] for i in crop_indices]
                work = WORK / year / name / window
                if work.exists():
                    shutil.rmtree(work)
                work.mkdir(parents=True, exist_ok=True)
                flags = speaking_on_crops(
                    crops,
                    taus[name][0],
                    taus[name][1],
                    work,
                    openface_bin=openface_bin,
                    openface_cwd=openface_cwd,
                    warmup_frames=DEFAULT_WINDOW_FRAMES,
                )
                for j, i in enumerate(crop_indices):
                    speaking[i] = int(flags[j]) if j < len(flags) else 0
            speaking_by_name[name] = speaking

        assert base_rows is not None
        rows = []
        for i, base in enumerate(base_rows):
            if not any(matched_by_name[n][i] for n in names):
                continue
            row = dict(base)
            for name in names:
                row[f"{name}_matched"] = matched_by_name[name][i]
                row[f"{name}_speaking"] = (
                    speaking_by_name[name][i] if matched_by_name[name][i] else 0
                )
            rows.append(row)
        df = pd.DataFrame(rows)
        out = OUT / f"{year}_new.csv"
        df.to_csv(out, index=False)
        sp_bits = ", ".join(
            f"{n}_speaking={int(df[f'{n}_speaking'].sum()) if len(df) else 0}"
            for n in names
        )
        print(f"Wrote {out.name}: {len(df)} rows (any matched), {sp_bits}", flush=True)

    clips = sorted(CLIPS.glob("source_*.mp4"))
    zip_path = OUT / "all_clips.zip"
    zip_path.unlink(missing_ok=True)
    if clips:
        subprocess.run(
            ["zip", "-q", "-j", str(zip_path), *[str(c) for c in clips]],
            check=True,
        )
    pd.DataFrame(
        [
            {
                "file": p.name,
                "year": p.name.split("_")[1],
                "window": p.stem.split("_", 2)[-1],
            }
            for p in clips
        ]
    ).to_csv(OUT / "clips_manifest.csv", index=False)
    print(f"\nALL DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
