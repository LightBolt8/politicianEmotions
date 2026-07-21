"""Subsample ~2k frames/candidate, run OpenFace with 2D landmarks, correlate
eye-span (landmark 36–45 distance) with year and expressivity.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
OF_BIN = Path.home() / "OpenFace/build/bin/FeatureExtraction"
OF_CWD = OF_BIN.parent.parent  # build/
OUT = ROOT / "new_plots" / "landmark_subsample"
VID_DIR = OUT / "videos"
CSV_DIR = OUT / "openface"
SAMPLE = 2000

# Same analysis set as main Spearmans (speaking for 2012+; full for 2004/08)
VIDEOS: list[tuple[str, int, str, Path]] = [
    ("Bush 2004", 2004, "2004", ROOT / "newExported/2004/Bush_clean_2004/Bush_clean_2004.mp4"),
    ("Kerry 2004", 2004, "2004", ROOT / "newExported/2004/Kerry_clean_2004/Kerry_clean_2004.mp4"),
    ("McCain 2008", 2008, "2008", ROOT / "newExported/2008/McCain_clean_2008/McCain_clean_2008.mp4"),
    ("Obama 2008", 2008, "2008", ROOT / "newExported/2008/Obama_clean_2008/Obama_clean_2008.mp4"),
    ("Romney 2012", 2012, "2012", ROOT / "newExported/2012/Romney_clean_2012/Romney_clean_2012_speaking.mp4"),
    ("Obama 2012", 2012, "2012", ROOT / "newExported/2012/Obama_clean_2012/Obama_clean_2012_speaking.mp4"),
    ("Trump 2016", 2016, "2016", ROOT / "newExported/2016/Trump_clean_2016/Trump_clean_2016_speaking.mp4"),
    ("Clinton 2016", 2016, "2016", ROOT / "newExported/2016/Clinton_clean_2016/Clinton_clean_2016_speaking.mp4"),
    ("Trump 2020", 2020, "2020", ROOT / "newExported/2020/Trump_clean_2020/Trump_clean_2020_speaking.mp4"),
    ("Biden 2020", 2020, "2020", ROOT / "newExported/2020/Biden_clean_2020/Biden_clean_2020_speaking.mp4"),
    ("Trump 2024 (1st)", 2024, "2024b", ROOT / "newExported/2024b/Trump_clean_2024b/Trump_clean_2024b_speaking.mp4"),
    ("Biden 2024", 2024, "2024b", ROOT / "newExported/2024b/Biden_clean_2024b/Biden_clean_2024b_speaking.mp4"),
    ("Trump 2024 (2nd)", 2024, "2024k", ROOT / "newExported/2024k/Trump_clean_2024/Trump_clean_2024_speaking.mp4"),
    ("Harris 2024", 2024, "2024k", ROOT / "newExported/2024k/Harris_clean_2024/Harris_clean_2024_speaking.mp4"),
]


def slug(label: str) -> str:
    return label.lower().replace(" ", "_").replace("(", "").replace(")", "")


def make_subsample_video(src: Path, dst: Path, n_sample: int = SAMPLE) -> int:
    """Evenly sample up to n_sample frames into a new mp4. Returns frames written."""
    if dst.is_file() and dst.stat().st_size > 0:
        cap = cv2.VideoCapture(str(dst))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n >= min(100, n_sample // 2):
            return n

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_keep = min(n_sample, total)
    idxs = set(np.linspace(0, total - 1, n_keep, dtype=int).tolist())

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    written = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in idxs:
            writer.write(frame)
            written += 1
        fi += 1
    cap.release()
    writer.release()
    return written


def run_openface(video: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{video.stem}.csv"
    if csv_path.is_file() and csv_path.stat().st_size > 1000:
        # already done
        return csv_path
    cmd = [
        str(OF_BIN),
        "-f",
        str(video.resolve()),
        "-out_dir",
        str(out_dir.resolve()),
        "-pose",
        "-aus",
        "-gaze",
        "-2Dfp",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(OF_CWD), capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.returncode != 0 or not csv_path.is_file():
        raise RuntimeError(
            f"OpenFace failed on {video.name} (exit {r.returncode})\n{r.stderr[-1500:]}"
        )
    print(f"  OpenFace {video.name}: {elapsed:.1f}s", flush=True)
    return csv_path


def eye_span_stats(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    need = {"x_36", "y_36", "x_45", "y_45"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing landmarks in {csv_path}")
    if "success" in df.columns:
        df = df[pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1]
    x36, y36 = df["x_36"].to_numpy(float), df["y_36"].to_numpy(float)
    x45, y45 = df["x_45"].to_numpy(float), df["y_45"].to_numpy(float)
    dist = np.sqrt((x45 - x36) ** 2 + (y45 - y36) ** 2)
    # normalize by face depth proxy if available (larger Tz = farther = smaller px span)
    row = {
        "n_landmark_frames": int(len(df)),
        "eye_span_px_mean": float(np.mean(dist)),
        "eye_span_px_median": float(np.median(dist)),
        "eye_span_px_sd": float(np.std(dist, ddof=1)) if len(dist) > 1 else 0.0,
    }
    if "pose_Tz" in df.columns:
        tz = pd.to_numeric(df["pose_Tz"], errors="coerce").to_numpy(float)
        # rough scale-normalized span: px * Tz (cancels distance if perspective ~1/z)
        norm = dist * np.abs(tz)
        row["eye_span_norm_mean"] = float(np.nanmean(norm))
    return row


def main() -> None:
    if not OF_BIN.is_file():
        raise FileNotFoundError(OF_BIN)
    OUT.mkdir(parents=True, exist_ok=True)
    VID_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    expr = pd.read_csv(ROOT / "newExported/aggression_with_AU23/candidate_summary.csv")
    expr_map = expr.set_index("candidate")["expressivity"].to_dict()

    rows = []
    t_all = time.time()
    for label, year, debate, src in VIDEOS:
        print(f"\n=== {label} ===", flush=True)
        if not src.is_file():
            print(f"  MISSING video {src}")
            continue
        sub = VID_DIR / f"{slug(label)}_sub{SAMPLE}.mp4"
        n_written = make_subsample_video(src, sub, SAMPLE)
        print(f"  subsample video: {n_written} frames → {sub.name}", flush=True)
        csv_path = run_openface(sub, CSV_DIR)
        stats_row = eye_span_stats(csv_path)
        stats_row.update(
            {
                "candidate": label,
                "year": year,
                "debate": debate,
                "expressivity": float(expr_map[label]),
                "source_video": str(src.relative_to(ROOT)),
            }
        )
        rows.append(stats_row)
        print(
            f"  eye_span_px={stats_row['eye_span_px_mean']:.2f}  "
            f"expr={stats_row['expressivity']:.3f}",
            flush=True,
        )

    out_df = pd.DataFrame(rows)
    out_csv = OUT / "eye_span_36_45_summary.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}", flush=True)

    print("\n=== Spearman correlations (candidate-level) ===", flush=True)
    for ycol in ["eye_span_px_mean", "eye_span_norm_mean"]:
        if ycol not in out_df.columns:
            continue
        for xcol in ["year", "expressivity"]:
            rho, p = stats.spearmanr(out_df[xcol], out_df[ycol])
            print(f"  {ycol} × {xcol}: ρ={rho:.3f}, p={p:.4f}", flush=True)

    print(f"\nTotal wall time: {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
