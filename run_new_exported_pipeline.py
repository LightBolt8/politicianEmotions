"""Rebuild speaking/AU analysis into newExported with dual-gate speaking filter.

Gate (2012+):
  Movement > Otsu(Movement) × --otsu-factor   (default 0.75)
  AND m > Otsu(m) × --m-otsu-factor            (default 0.85)
  then turn consolidation (gap≤25, min≥25).

Steps:
  1. Mirror Exported → newExported (hardlink clean mp4 + full OpenFace CSV)
  2. filter_speaking on each post-2008 clean video
  3. OpenFace on *_speaking.mp4
  4. analyze_aus + plot_au_comparison
  5. Spearman year × AU04/05/07 / aggression (speaking frames only)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from filter_speaking import filter_video

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (folder_year, candidate_folder_stem) under Exported/{folder}/
SPEAKING_YEARS: list[tuple[str, list[str]]] = [
    ("2012", ["Obama_clean_2012", "Romney_clean_2012"]),
    ("2016", ["Trump_clean_2016", "Clinton_clean_2016"]),
    ("2020", ["Trump_clean_2020", "Biden_clean_2020"]),
    ("2024b", ["Trump_clean_2024b", "Biden_clean_2024b"]),
    ("2024k", ["Trump_clean_2024", "Harris_clean_2024"]),
]

# Pre-2009: no speaking filter; hardlink full clean for comparison plots.
FULL_YEARS: list[tuple[str, list[str]]] = [
    ("2004", ["Bush_clean_2004", "Kerry_clean_2004"]),
    ("2008", ["McCain_clean_2008", "Obama_clean_2008"]),
]

ANGER = ("AU04_r", "AU05_r", "AU07_r")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def mirror_candidate(src_dir: Path, dst_dir: Path) -> Path:
    """Hardlink clean mp4 + CSV into dst; return clean video path."""
    stem = src_dir.name
    video = src_dir / f"{stem}.mp4"
    csv = src_dir / f"{stem}.csv"
    if not video.is_file():
        raise FileNotFoundError(video)
    if not csv.is_file():
        raise FileNotFoundError(csv)
    dst_dir.mkdir(parents=True, exist_ok=True)
    hardlink_or_copy(video, dst_dir / video.name)
    hardlink_or_copy(csv, dst_dir / csv.name)
    return dst_dir / video.name


def setup_tree(src_root: Path, dst_root: Path) -> list[Path]:
    """Mirror all candidates; return post-2008 clean videos to re-filter."""
    speaking_set = {(y, s) for y, stems in SPEAKING_YEARS for s in stems}
    speaking_videos: list[Path] = []
    for year, stems in FULL_YEARS + SPEAKING_YEARS:
        for stem in stems:
            src = src_root / year / stem
            dst = dst_root / year / stem
            video = mirror_candidate(src, dst)
            if (year, stem) in speaking_set:
                speaking_videos.append(video)
    return speaking_videos


def filter_all(
    videos: list[Path],
    *,
    otsu_factor: float,
    m_otsu_factor: float,
) -> None:
    for video in videos:
        print(f"\n=== Speaking filter: {video} ===", flush=True)
        filter_video(
            video,
            otsu_factor=otsu_factor,
            m_otsu_factor=m_otsu_factor,
            write_deleted=True,
            use_turns=True,
        )


def openface_speaking(dst_root: Path) -> None:
    """OpenFace only on new speaking clips (2004/2008 keep hardlinked full CSVs)."""
    speaking = sorted(dst_root.rglob("*_speaking.mp4"))
    speaking = [p for p in speaking if "time_windows" not in p.parts]
    if not speaking:
        raise FileNotFoundError(f"No *_speaking.mp4 under {dst_root}")
    cmd = [
        PY,
        "run_openface.py",
        "--force",
        "--openface-bin",
        str(Path.home() / "OpenFace/build/bin/FeatureExtraction"),
        *[str(p) for p in speaking],
    ]
    run(cmd)


def analyze_and_compare(dst_root: Path) -> None:
    run([PY, "analyze_aus.py", "--all", "--data-dir", str(dst_root)])


def discover_speaking_csvs(root: Path) -> list[tuple[int, str, Path]]:
    rows: list[tuple[int, str, Path]] = []
    for csv in sorted(root.rglob("*_speaking.csv")):
        if "time_windows" in csv.parts:
            continue
        stem = csv.stem
        if not stem.endswith("_speaking"):
            continue
        base = stem[: -len("_speaking")]
        if "_clean_" not in base or csv.parent.name != base:
            continue
        candidate, year_s = base.rsplit("_clean_", 1)
        year_key = int(re.match(r"(\d+)", year_s).group(1)) if re.match(r"(\d+)", year_s) else 0
        if year_key <= 2008:
            continue
        rows.append((year_key, candidate, csv))
    return rows


def run_spearman(dst_root: Path) -> None:
    out = dst_root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for year_key, candidate, csv in discover_speaking_csvs(dst_root):
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        if "success" in df.columns:
            df = df[pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1]
        missing = [c for c in ANGER if c not in df.columns]
        if missing:
            print(f"skip {csv}: missing {missing}")
            continue
        sub = df[list(ANGER)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        sub = sub.assign(
            year=year_key,
            candidate=candidate,
            aggression=sub[list(ANGER)].mean(axis=1),
        )
        frames.append(sub)
        print(f"Spearman load {candidate} {year_key}: {len(sub)} frames")

    if not frames:
        raise FileNotFoundError(f"No speaking CSVs under {dst_root}")

    all_df = pd.concat(frames, ignore_index=True)
    measures = [
        ("AU04", "AU04_r"),
        ("AU05", "AU05_r"),
        ("AU07", "AU07_r"),
        ("aggression", "aggression"),
    ]
    rows = []
    for label, col in measures:
        rho, p = stats.spearmanr(all_df["year"], all_df[col])
        rows.append(
            {
                "span": "2012–2024 speaking",
                "measure": label,
                "rho": float(rho),
                "p": float(p),
                "n": int(len(all_df)),
            }
        )
        print(f"  {label}: ρ={rho:.3f} p={p:.2e} n={len(all_df)}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "spearman_year_au_aggression.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis("off")
    ax.set_title(
        "Spearman: year × AU / aggression (speaking frames, 2012–2024)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    table = ax.table(
        cellText=[
            [
                r["span"],
                r["measure"],
                f"{r['rho']:.3f}",
                f"{r['p']:.2e}",
                f"{int(r['n']):,}",
            ]
            for _, r in summary.iterrows()
        ],
        colLabels=["span", "measure", "ρ", "p", "n"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.6)
    fig.savefig(out / "spearman_year_au_aggression.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'spearman_year_au_aggression.png'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=ROOT / "Exported")
    p.add_argument("--dst", type=Path, default=ROOT / "newExported")
    p.add_argument("--otsu-factor", type=float, default=0.75, help="Movement Otsu factor")
    p.add_argument("--m-otsu-factor", type=float, default=0.85, help="m intensity Otsu factor")
    p.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip hardlink mirror (reuse existing newExported tree).",
    )
    p.add_argument("--skip-filter", action="store_true")
    p.add_argument("--skip-openface", action="store_true")
    p.add_argument("--skip-analyze", action="store_true")
    p.add_argument("--skip-spearman", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    t0 = time.time()

    print(
        f"newExported pipeline: mov×{args.otsu_factor:g} AND m×{args.m_otsu_factor:g}\n"
        f"  src={src}\n  dst={dst}",
        flush=True,
    )

    if not args.skip_setup:
        print("\n=== 1. Mirror Exported → newExported (hardlinks) ===", flush=True)
        videos = setup_tree(src, dst)
    else:
        videos = []
        for year, stems in SPEAKING_YEARS:
            for stem in stems:
                videos.append(dst / year / stem / f"{stem}.mp4")

    if not args.skip_filter:
        print("\n=== 2. Speaking / nonspeaking filter ===", flush=True)
        filter_all(videos, otsu_factor=args.otsu_factor, m_otsu_factor=args.m_otsu_factor)

    if not args.skip_openface:
        print("\n=== 3. OpenFace on speaking videos ===", flush=True)
        openface_speaking(dst)

    if not args.skip_analyze:
        print("\n=== 4. Per-candidate plots + comparison ===", flush=True)
        analyze_and_compare(dst)

    if not args.skip_spearman:
        print("\n=== 5. Spearman ===", flush=True)
        run_spearman(dst)

    elapsed = time.time() - t0
    print(f"\nALL DONE in {elapsed / 60:.1f} min → {dst}", flush=True)


if __name__ == "__main__":
    main()
