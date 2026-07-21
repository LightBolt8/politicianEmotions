"""After main landmark run: 2020 Trump/Biden, 1000 frames, compare camera
intrinsics default (~233.3,233.3,128,128) vs (256,256,128,128).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OF_BIN = Path.home() / "OpenFace/build/bin/FeatureExtraction"
OF_CWD = OF_BIN.parent.parent
OUT = ROOT / "new_plots" / "landmark_subsample"
VID_DIR = OUT / "videos"
CAM_DIR = OUT / "camera_compare_2020"
SAMPLE = 1000

MAIN_SUMMARY = OUT / "eye_span_36_45_summary.csv"
MAIN_LOG = OUT / "run.log"

VIDEOS = [
    ("Trump 2020", ROOT / "newExported/2020/Trump_clean_2020/Trump_clean_2020_speaking.mp4"),
    ("Biden 2020", ROOT / "newExported/2020/Biden_clean_2020/Biden_clean_2020_speaking.mp4"),
]


def wait_for_main(timeout_s: float = 6 * 3600) -> None:
    t0 = time.time()
    print("Waiting for main landmark run to finish...", flush=True)
    while time.time() - t0 < timeout_s:
        if MAIN_SUMMARY.is_file() and MAIN_SUMMARY.stat().st_size > 100:
            # also ensure no FeatureExtraction still running on main videos
            r = subprocess.run(["pgrep", "-x", "FeatureExtraction"], capture_output=True)
            if r.returncode != 0:
                print("Main run complete.", flush=True)
                return
            # FeatureExtraction might be ours for camera compare later — only wait if
            # summary not written with all 14 rows yet
            try:
                n = len(pd.read_csv(MAIN_SUMMARY))
                if n >= 14:
                    # wait until OF free
                    while subprocess.run(["pgrep", "-x", "FeatureExtraction"], capture_output=True).returncode == 0:
                        time.sleep(10)
                    print("Main run complete.", flush=True)
                    return
            except Exception:
                pass
        time.sleep(20)
    raise TimeoutError("Timed out waiting for main landmark run")


def make_subsample(src: Path, dst: Path, n_sample: int = SAMPLE) -> int:
    if dst.is_file() and dst.stat().st_size > 0:
        cap = cv2.VideoCapture(str(dst))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n >= min(100, n_sample // 2):
            return n
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_keep = min(n_sample, total)
    idxs = set(np.linspace(0, total - 1, n_keep, dtype=int).tolist())
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = fi = 0
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


def run_of(video: Path, out_dir: Path, tag: str, cam: tuple[float, float, float, float] | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # OpenFace names output after input stem; use unique out dirs per cam setting
    csv_path = out_dir / f"{video.stem}.csv"
    details = out_dir / f"{video.stem}_of_details.txt"
    if csv_path.is_file() and csv_path.stat().st_size > 1000:
        return csv_path
    cmd = [
        str(OF_BIN),
        "-f", str(video.resolve()),
        "-out_dir", str(out_dir.resolve()),
        "-pose", "-aus", "-gaze", "-2Dfp",
    ]
    if cam is not None:
        fx, fy, cx, cy = cam
        cmd.extend(["-fx", str(fx), "-fy", str(fy), "-cx", str(cx), "-cy", str(cy)])
    print(f"  OpenFace [{tag}] ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(OF_CWD), capture_output=True, text=True)
    print(f"  done in {time.time()-t0:.1f}s (exit {r.returncode})", flush=True)
    if r.returncode != 0 or not csv_path.is_file():
        raise RuntimeError(f"OpenFace failed [{tag}]: {r.stderr[-1200:]}")
    return csv_path


def summarize(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "success" in df.columns:
        df = df[pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1]
    dist = np.sqrt(
        (df["x_45"] - df["x_36"]) ** 2 + (df["y_45"] - df["y_36"]) ** 2
    )
    out = {
        "n": len(df),
        "eye_span_px_mean": float(dist.mean()),
        "eye_span_px_median": float(dist.median()),
        "eye_span_px_sd": float(dist.std(ddof=1)) if len(dist) > 1 else 0.0,
    }
    if "pose_Tz" in df.columns:
        tz = pd.to_numeric(df["pose_Tz"], errors="coerce")
        out["pose_Tz_mean"] = float(tz.mean())
        out["eye_span_norm_mean"] = float((dist * tz.abs()).mean())
    # quick expressivity proxy from this subsample (excl AU25/26/45)
    drop = {"AU25_r", "AU26_r", "AU45_r"}
    au_r = [c for c in df.columns if c.endswith("_r") and c.startswith("AU") and c not in drop]
    if au_r:
        out["expressivity_r_mean"] = float(df[au_r].apply(pd.to_numeric, errors="coerce").fillna(0).mean().mean())
    aggr = [c for c in ("AU04_r", "AU05_r", "AU07_r", "AU23_r") if c in df.columns]
    if aggr:
        out["aggression_4_mean"] = float(df[aggr].apply(pd.to_numeric, errors="coerce").fillna(0).mean().mean())
    # read camera from details if present
    details = csv_path.with_name(csv_path.stem + "_of_details.txt")
    if details.is_file():
        for line in details.read_text().splitlines():
            if line.startswith("Camera parameters:"):
                out["camera_params"] = line.split(":", 1)[1].strip()
    return out


def main() -> None:
    wait_for_main()
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    cams = {
        "default_233": None,  # OpenFace default for 256px → 233.333,233.333,128,128
        "fxfy256": (256.0, 256.0, 128.0, 128.0),
    }
    for label, src in VIDEOS:
        print(f"\n=== {label} ===", flush=True)
        slug = label.lower().replace(" ", "_")
        sub = VID_DIR / f"{slug}_sub{SAMPLE}.mp4"
        n = make_subsample(src, sub, SAMPLE)
        print(f"  subsample: {n} frames → {sub.name}", flush=True)
        for tag, cam in cams.items():
            out_dir = CAM_DIR / tag / slug
            csv_path = run_of(sub, out_dir, f"{label} {tag}", cam)
            stats = summarize(csv_path)
            stats.update({"candidate": label, "camera_setting": tag})
            rows.append(stats)
            print(
                f"  [{tag}] eye_span={stats['eye_span_px_mean']:.3f}  "
                f"Tz={stats.get('pose_Tz_mean', float('nan')):.2f}  "
                f"expr={stats.get('expressivity_r_mean', float('nan')):.4f}  "
                f"cam={stats.get('camera_params', '?')}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    out_csv = CAM_DIR / "camera_compare_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}", flush=True)

    # side-by-side delta
    print("\n=== Default vs 256,256,128,128 ===", flush=True)
    for label in [r[0] for r in VIDEOS]:
        a = df[(df.candidate == label) & (df.camera_setting == "default_233")].iloc[0]
        b = df[(df.candidate == label) & (df.camera_setting == "fxfy256")].iloc[0]
        print(f"\n{label}:", flush=True)
        for col in [
            "eye_span_px_mean",
            "eye_span_px_median",
            "pose_Tz_mean",
            "eye_span_norm_mean",
            "expressivity_r_mean",
            "aggression_4_mean",
        ]:
            if col in a and col in b and pd.notna(a[col]) and pd.notna(b[col]):
                d = float(b[col]) - float(a[col])
                print(
                    f"  {col}: default={a[col]:.4f}  fxfy256={b[col]:.4f}  Δ={d:+.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
