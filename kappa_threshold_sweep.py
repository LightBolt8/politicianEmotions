"""Sweep mov/m Otsu factors on validation window crops; report Cohen's κ.

Requires newExported/time_windows/work/{year}/{name}/{window}/crops.csv
from export_time_windows.py, plus labeled testingSpeak - {year}_new.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from filter_speaking import (
    DEFAULT_MAX_GAP_FRAMES,
    DEFAULT_MIN_TURN_FRAMES,
    DEFAULT_WINDOW_FRAMES,
    compute_m,
    consolidate_turns,
    otsu_threshold,
    speaking_mask,
)

ROOT = Path(__file__).resolve().parent
TW = ROOT / "newExported" / "time_windows"
WORK = TW / "work"
DATA = ROOT / "newExported"

PAIRS = {
    "2012": [("Obama", "Obama_clean_2012", "Obama_actual"), ("Romney", "Romney_clean_2012", "Romney_actual")],
    "2016": [("Trump", "Trump_clean_2016", "Trump_actual"), ("Clinton", "Clinton_clean_2016", "Clinton_actual")],
    "2020": [("Trump", "Trump_clean_2020", "Trump_actual"), ("Biden", "Biden_clean_2020", "Biden_actual")],
    "2024b": [("Trump", "Trump_clean_2024b", "Trump_actual"), ("Biden", "Biden_clean_2024b", "Biden_actual")],
    "2024k": [("Trump", "Trump_clean_2024", "Trump_actual"), ("Harris", "Harris_clean_2024", "Harris_actual")],
}

MOV_FACTORS = [0.70, 0.75, 0.80, 0.85, 0.90]
M_FACTORS = [0.70, 0.75, 0.80, 0.85, 0.90, None]  # None = movement-only


def kappa(a: np.ndarray, s: np.ndarray) -> float:
    a = np.asarray(a, dtype=int)
    s = np.asarray(s, dtype=int)
    if len(a) == 0:
        return float("nan")
    po = float((a == s).mean())
    classes = sorted(set(a) | set(s))
    pe = sum(float((a == c).mean() * (s == c).mean()) for c in classes)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def taus_from_clean(clean_csv: Path, mov_f: float, m_f: float | None) -> tuple[float, float | None]:
    df = pd.read_csv(clean_csv)
    df.columns = [c.strip() for c in df.columns]
    success = None
    if "success" in df.columns:
        success = pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1
    *_r, tau_mov = speaking_mask(df, otsu_factor=mov_f, ref_mask=success)
    tau_m = None
    if m_f is not None:
        tau_m = float(otsu_threshold(compute_m(df), ref_mask=success) * m_f)
    return float(tau_mov), tau_m


def speaking_from_crops(
    crops_csv: Path,
    tau_mov: float,
    tau_m: float | None,
    *,
    warmup: int = DEFAULT_WINDOW_FRAMES,
) -> list[int]:
    df = pd.read_csv(crops_csv)
    df.columns = [c.strip() for c in df.columns]
    # crops.csv includes warmup pad at the start (same as export)
    keep, m, *_rest = speaking_mask(
        df,
        window_frames=DEFAULT_WINDOW_FRAMES,
        mode="absolute",
        min_movement=tau_mov,
    )
    if tau_m is not None:
        keep = keep & (m.fillna(0.0) > float(tau_m))
    keep, _ = consolidate_turns(
        keep.fillna(False),
        max_gap_frames=DEFAULT_MAX_GAP_FRAMES,
        min_turn_frames=DEFAULT_MIN_TURN_FRAMES,
    )
    flags = keep.astype(int).tolist()
    flags = flags[warmup:]
    return flags


def pred_speaking_for_candidate(
    year: str,
    name: str,
    stem: str,
    lab: pd.DataFrame,
    mov_f: float,
    m_f: float | None,
) -> np.ndarray:
    """Rebuild speaking vector aligned to lab rows (0 if unmatched)."""
    clean = DATA / year / stem / f"{stem}.csv"
    tau_mov, tau_m = taus_from_clean(clean, mov_f, m_f)
    out = np.zeros(len(lab), dtype=int)
    mcol = f"{name}_matched"
    for window, wdf in lab.groupby("window", sort=False):
        idxs = wdf.index[wdf[mcol] == 1].tolist()
        if not idxs:
            continue
        crops = WORK / year / name / str(window) / "crops.csv"
        if not crops.is_file():
            raise FileNotFoundError(crops)
        flags = speaking_from_crops(crops, tau_mov, tau_m)
        if len(flags) < len(idxs):
            flags = flags + [0] * (len(idxs) - len(flags))
        for j, i in enumerate(idxs):
            out[i] = int(flags[j])
    return out


def main() -> None:
    # verify crops exist
    missing = []
    for year, pairs in PAIRS.items():
        for name, stem, _act in pairs:
            for window_dir in (WORK / year / name).glob("*"):
                if window_dir.is_dir() and not (window_dir / "crops.csv").is_file():
                    missing.append(str(window_dir))
            if not (WORK / year / name).is_dir():
                missing.append(f"{year}/{name}")
    if missing:
        raise SystemExit(
            f"Missing crop CSVs under {WORK} ({len(missing)}). "
            "Re-run export_time_windows.py first.\n"
            + "\n".join(missing[:10])
        )

    grid_rows = []
    detail_rows = []

    for mov_f in MOV_FACTORS:
        for m_f in M_FACTORS:
            tag = f"mov×{mov_f:g}" + (f" AND m×{m_f:g}" if m_f is not None else " (mov-only)")
            ks = []
            fp = fn = 0
            for year, pairs in PAIRS.items():
                lab_path = TW / f"testingSpeak - {year}_new.csv"
                lab = pd.read_csv(lab_path)
                lab.columns = [c.strip() for c in lab.columns]
                for name, stem, act in pairs:
                    pred = pred_speaking_for_candidate(year, name, stem, lab, mov_f, m_f)
                    mask = (lab[f"{name}_matched"] == 1) & lab[act].notna()
                    a = pd.to_numeric(lab.loc[mask, act], errors="coerce").astype(int).to_numpy()
                    s = pred[mask.to_numpy()]
                    k = kappa(a, s)
                    ks.append(k)
                    fp += int(((a == 0) & (s == 1)).sum())
                    fn += int(((a == 1) & (s == 0)).sum())
                    detail_rows.append(
                        {
                            "mov_factor": mov_f,
                            "m_factor": m_f if m_f is not None else "",
                            "rule": tag,
                            "year": year,
                            "candidate": name,
                            "kappa": k,
                            "n": len(a),
                            "FP": int(((a == 0) & (s == 1)).sum()),
                            "FN": int(((a == 1) & (s == 0)).sum()),
                        }
                    )
            mean_k = float(np.nanmean(ks))
            grid_rows.append(
                {
                    "mov_factor": mov_f,
                    "m_factor": m_f if m_f is not None else "",
                    "rule": tag,
                    "mean_kappa": mean_k,
                    "FP": fp,
                    "FN": fn,
                    "errors": fp + fn,
                }
            )
            print(f"{tag:28} meanκ={mean_k:.3f}  FP={fp} FN={fn} err={fp+fn}")

    grid = pd.DataFrame(grid_rows).sort_values("mean_kappa", ascending=False)
    detail = pd.DataFrame(detail_rows)
    grid.to_csv(TW / "kappa_threshold_grid.csv", index=False)
    detail.to_csv(TW / "kappa_threshold_detail.csv", index=False)
    print("\nTop 10 by mean κ:")
    print(grid.head(10).to_string(index=False))
    print(f"\nWrote {TW / 'kappa_threshold_grid.csv'}")
    print(f"Wrote {TW / 'kappa_threshold_detail.csv'}")


if __name__ == "__main__":
    main()
