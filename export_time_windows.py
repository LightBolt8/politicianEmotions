"""Speaking-validation export with source-frame realignment.

Rebuilds the missing preprocess source-frame index (ArcFace rematch, no
video rewrite), maps each clean OpenFace speaking label onto Dataset time,
then samples 1 fps in windows 10–11 / 40–41 / 70–71.

Output: Exported/time_windows/
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

from filter_speaking import consolidate_turns, speaking_mask
from preprocessing_data import (
    build_known_embeddings,
    is_frontal,
    match_candidate,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Exported" / "time_windows"
CLIPS = OUT / "clips"

OTSU_FACTOR = 0.9
SKIP = 30  # ~1 fps sample in export CSV
PREPROCESS_SKIP = 6
THRESHOLD = 0.6
MAX_YAW = 30.0
WINDOWS = (
    ("10-11min", 10 * 60.0, 11 * 60.0),
    ("40-41min", 40 * 60.0, 41 * 60.0),
    ("70-71min", 70 * 60.0, 71 * 60.0),
)

# (year, dataset stem, (A, B), start_seconds, clean CSVs, ref_a, ref_b)
# 2012 first so the in-flight rematch cache is reused immediately.
# 2004/2008 skipped (no speaking reanalysis).
YEARS = (
    (
        "2012",
        "2012",
        ("Obama", "Romney"),
        0.0,
        (
            ROOT / "Exported/2012/Obama_clean_2012/Obama_clean_2012.csv",
            ROOT / "Exported/2012/Romney_clean_2012/Romney_clean_2012.csv",
        ),
        ROOT / "refs/2012/candidate_A.jpg",
        ROOT / "refs/2012/candidate_B.jpg",
    ),
    (
        "2016",
        "2016",
        ("Trump", "Clinton"),
        0.0,
        (
            ROOT / "Exported/2016/Trump_clean_2016/Trump_clean_2016.csv",
            ROOT / "Exported/2016/Clinton_clean_2016/Clinton_clean_2016.csv",
        ),
        ROOT / "refs/2016/candidate_A.jpg",
        ROOT / "refs/2016/candidate_B.jpg",
    ),
    (
        "2020",
        "2020",
        ("Trump", "Biden"),
        400.0,
        (
            ROOT / "Exported/2020/Trump_clean_2020/Trump_clean_2020.csv",
            ROOT / "Exported/2020/Biden_clean_2020/Biden_clean_2020.csv",
        ),
        ROOT / "refs/2020/candidate_A.jpg",
        ROOT / "refs/2020/candidate_B.jpg",
    ),
    (
        "2024b",
        "2024b",
        ("Trump", "Biden"),
        11.0,
        (
            ROOT / "Exported/2024b/Trump_clean_2024b/Trump_clean_2024b.csv",
            ROOT / "Exported/2024b/Biden_clean_2024b/Biden_clean_2024b.csv",
        ),
        ROOT / "refs/2024b/candidate_A.jpg",
        ROOT / "refs/2024b/candidate_B.jpg",
    ),
    (
        "2024k",
        "2024k",
        ("Trump", "Harris"),
        0.0,
        (
            ROOT / "Exported/2024k/Trump_clean_2024/Trump_clean_2024.csv",
            ROOT / "Exported/2024k/Harris_clean_2024/Harris_clean_2024.csv",
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


def ffmpeg_cut(dataset: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 1000:
        print(f"skip existing {dst.name}")
        return
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{end - start:.3f}",
        "-i", str(dataset),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        str(dst),
    ]
    print(f"cut {dataset.name} [{start:.0f}-{end:.0f}s] -> {dst.name}", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)


def create_face_app() -> FaceAnalysis:
    # CPU was faster than CoreML in local benchmarks (~1.7h vs ~2h for 2012).
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def map_paths(clean_csv: Path) -> tuple[Path, Path, Path]:
    """source_frames.npy + meta.json + in-progress checkpoint beside the clean CSV."""
    base = clean_csv.with_name(clean_csv.stem + "_source_frames")
    return base.with_suffix(".npy"), base.with_suffix(".json"), base.with_suffix(".ckpt.npz")


def load_or_build_source_maps(
    dataset: Path,
    names: tuple[str, str],
    cleans: tuple[Path, Path],
    ref_a: Path,
    ref_b: Path,
    start_seconds: float,
    expected_counts: tuple[int, int],
    *,
    skip_rate: int = PREPROCESS_SKIP,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Replay preprocess matching; return {name: source_frame_index per clean row}."""
    npy_a, meta_a, ckpt_a = map_paths(cleans[0])
    npy_b, meta_b, ckpt_b = map_paths(cleans[1])
    # One shared checkpoint (A path); B path unused but kept for symmetry.
    ckpt_path = ckpt_a

    if (
        not force
        and npy_a.is_file()
        and npy_b.is_file()
        and meta_a.is_file()
        and meta_b.is_file()
    ):
        fa = np.load(npy_a)
        fb = np.load(npy_b)
        if len(fa) == expected_counts[0] and len(fb) == expected_counts[1]:
            print(
                f"Loaded source maps: {names[0]}={len(fa)}, {names[1]}={len(fb)}",
                flush=True,
            )
            return {names[0]: fa.astype(np.int32), names[1]: fb.astype(np.int32)}
        print(
            f"Cached maps length mismatch "
            f"({len(fa)}/{len(fb)} vs {expected_counts}); rebuilding...",
            flush=True,
        )

    print("Building source-frame maps via ArcFace rematch (no video write)...", flush=True)
    app = create_face_app()
    embeddings = build_known_embeddings(app, ref_a, ref_b)

    cap = cv2.VideoCapture(str(dataset))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {dataset}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_count = int(start_seconds * source_fps)
    frames_a: list[int] = []
    frames_b: list[int] = []

    if ckpt_path.is_file() and not force:
        ck = np.load(ckpt_path)
        frame_count = int(ck["frame_count"])
        frames_a = ck["frames_a"].astype(np.int32).tolist()
        frames_b = ck["frames_b"].astype(np.int32).tolist()
        print(
            f"Resuming rematch at frame {frame_count} "
            f"({names[0]}={len(frames_a)}, {names[1]}={len(frames_b)})",
            flush=True,
        )

    if frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    progress_every = max(skip_rate * 500, 3000)
    checkpoint_every = 9000  # ~5 min of source video

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip_rate != 0:
            continue
        if progress_every and frame_count % progress_every < skip_rate:
            print(
                f"  rematch frame {frame_count}/{total} "
                f"({names[0]}={len(frames_a)}, {names[1]}={len(frames_b)})",
                flush=True,
            )

        for face in app.get(frame):
            if not is_frontal(face, MAX_YAW):
                continue
            idx = match_candidate(face.embedding, embeddings, THRESHOLD)
            if idx == 0:
                frames_a.append(frame_count)
            elif idx == 1:
                frames_b.append(frame_count)

        if frame_count % checkpoint_every == 0:
            np.savez(
                ckpt_path,
                frame_count=frame_count,
                frames_a=np.asarray(frames_a, dtype=np.int32),
                frames_b=np.asarray(frames_b, dtype=np.int32),
            )

    cap.release()

    arr_a = np.asarray(frames_a, dtype=np.int32)
    arr_b = np.asarray(frames_b, dtype=np.int32)
    np.save(npy_a, arr_a)
    np.save(npy_b, arr_b)
    meta = {
        "dataset": str(dataset),
        "start_seconds": start_seconds,
        "skip_rate": skip_rate,
        "threshold": THRESHOLD,
        "max_yaw": MAX_YAW,
        "ref_a": str(ref_a),
        "ref_b": str(ref_b),
        "count_a": int(len(arr_a)),
        "count_b": int(len(arr_b)),
        "expected_a": int(expected_counts[0]),
        "expected_b": int(expected_counts[1]),
    }
    meta_a.write_text(json.dumps(meta, indent=2))
    meta_b.write_text(json.dumps(meta, indent=2))
    ckpt_path.unlink(missing_ok=True)
    ckpt_b.unlink(missing_ok=True)
    print(
        f"Saved maps: {names[0]}={len(arr_a)} (expected {expected_counts[0]}), "
        f"{names[1]}={len(arr_b)} (expected {expected_counts[1]})",
        flush=True,
    )
    if len(arr_a) != expected_counts[0] or len(arr_b) != expected_counts[1]:
        print(
            "WARNING: rematch counts != clean CSV lengths. "
            "Alignment will truncate to min length; check start_seconds/refs.",
            flush=True,
        )
    return {names[0]: arr_a, names[1]: arr_b}


def speaking_by_source_frame(
    clean_csv: Path,
    source_frames: np.ndarray,
) -> dict[int, int]:
    """Map source_frame -> speaking 0/1 using clean OpenFace + Otsu×0.9."""
    df = pd.read_csv(clean_csv)
    df.columns = [c.strip() for c in df.columns]
    success = None
    if "success" in df.columns:
        success = pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1
    keep, *_ = speaking_mask(df, otsu_factor=OTSU_FACTOR, ref_mask=success)
    if success is not None:
        keep = keep & success
    keep, _ = consolidate_turns(keep)
    speaking = keep.fillna(False).astype(np.int8).to_numpy()

    n = min(len(speaking), len(source_frames))
    out: dict[int, int] = {}
    for src, sp in zip(source_frames[:n], speaking[:n], strict=True):
        # Last write wins if multiple faces matched same source frame.
        out[int(src)] = int(sp)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--years",
        nargs="+",
        metavar="YEAR",
        help="Only process these years (e.g. 2024b 2024k). Default: all in YEARS.",
    )
    p.add_argument(
        "--maps-only",
        action="store_true",
        help="Only rebuild source-frame maps (rematch); skip CSV/clip export.",
    )
    p.add_argument(
        "--force-remap",
        action="store_true",
        help="Rebuild maps even if cached .npy files already match CSV lengths.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)

    selected = list(YEARS)
    if args.years:
        want = set(args.years)
        selected = [y for y in YEARS if y[0] in want]
        missing = want - {y[0] for y in selected}
        if missing:
            raise SystemExit(f"Unknown year(s): {sorted(missing)}")

    for year, stem, names, offset, cleans, ref_a, ref_b in selected:
        dataset = ROOT / "Dataset" / f"{stem}.mp4"
        if not dataset.is_file():
            print(f"SKIP {year}: missing dataset")
            continue
        if not all(c.is_file() for c in cleans):
            print(f"SKIP {year}: missing clean CSV")
            continue

        n_src, fps, src_dur = source_props(dataset)
        print(f"\n=== {year} ({src_dur/60:.1f} min) ===", flush=True)

        expected = tuple(sum(1 for _ in open(c, encoding="utf-8")) - 1 for c in cleans)
        maps = load_or_build_source_maps(
            dataset,
            names,
            cleans,
            ref_a,
            ref_b,
            offset,
            expected,  # type: ignore[arg-type]
            force=args.force_remap,
        )
        if args.maps_only:
            print(f"{year}: maps only — skip CSV/clips", flush=True)
            continue

        luts = {
            name: speaking_by_source_frame(clean, maps[name])
            for name, clean in zip(names, cleans, strict=True)
        }

        rows: dict[str, list[dict]] = {n: [] for n in names}

        for window, t0, t1 in WINDOWS:
            if t0 >= src_dur:
                continue
            end = min(t1, src_dur)
            clip = CLIPS / f"source_{year}_{window}.mp4"
            ffmpeg_cut(dataset, t0, end, clip)

            f0 = int(np.ceil(t0 * fps / SKIP) * SKIP)
            f1 = int(np.floor((end * fps - 1e-9) / SKIP) * SKIP)
            for src_frame in range(f0, f1 + 1, SKIP):
                ts = src_frame / fps
                clip_ts = ts - t0
                for name in names:
                    sp = luts[name].get(src_frame)
                    matched = 0 if sp is None else 1
                    rows[name].append(
                        {
                            "year": year,
                            "candidate": name,
                            "window": window,
                            "clip_file": clip.name,
                            "frame": src_frame,
                            "timestamp": round(ts, 6),
                            "timestamp_mmss": mmss(ts),
                            "clip_timestamp": round(clip_ts, 6),
                            "matched": matched,
                            "speaking": 0 if sp is None else sp,
                        }
                    )

        for name in names:
            df = pd.DataFrame(rows[name])
            out = OUT / f"{year}_{name}.csv"
            df.to_csv(out, index=False)
            m = int(df["matched"].sum())
            s = int(df["speaking"].sum())
            print(
                f"Wrote {out.name}: {len(df)} rows, matched={m}, speaking={s}",
                flush=True,
            )
            for w, g in df.groupby("window"):
                print(
                    f"  {w}: matched={int(g.matched.sum())}/{len(g)}, "
                    f"speaking={int(g.speaking.sum())}",
                    flush=True,
                )

    if not args.maps_only:
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
