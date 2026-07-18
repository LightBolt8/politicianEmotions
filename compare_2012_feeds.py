"""Compare 2012 split-screen vs switched-feed OpenFace AUs (full tracks, no speaking filter)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DROP_EXPR = {"AU45_r"}
AGGR = ("AU04_r", "AU05_r", "AU07_r")

DEFAULT_PATHS = {
    ("Obama", "split"): ROOT / "Exported/2012/Obama_clean_2012/Obama_clean_2012.csv",
    ("Romney", "split"): ROOT / "Exported/2012/Romney_clean_2012/Romney_clean_2012.csv",
    ("Obama", "switched"): ROOT
    / "Exported/2012switched/Obama_clean_2012switched/Obama_clean_2012switched.csv",
    ("Romney", "switched"): ROOT
    / "Exported/2012switched/Romney_clean_2012switched/Romney_clean_2012switched.csv",
}


def load_aus(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "success" in df.columns:
        df = df[pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1]
    au_cols = [c for c in df.columns if re.fullmatch(r"AU\d{2}_r", c)]
    return df[au_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def summarize(paths: dict[tuple[str, str], Path]) -> pd.DataFrame:
    rows = []
    for (cand, feed), path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        aus = load_aus(path)
        expr_cols = [c for c in aus.columns if c not in DROP_EXPR]
        expr = float(aus[expr_cols].mean().mean())
        aggr = float(aus[list(AGGR)].mean().mean())
        row = {
            "candidate": cand,
            "feed": feed,
            "n_frames": len(aus),
            "expressivity": expr,
            "aggression": aggr,
            "m_AU25_AU26": float((aus["AU25_r"] + aus["AU26_r"]).mean()),
        }
        for au in sorted(
            [c for c in aus.columns if c not in DROP_EXPR],
            key=lambda x: int(x[2:4]),
        ):
            row[au.replace("_r", "")] = float(aus[au].mean())
        rows.append(row)
        print(f"{cand:7s} {feed:9s}: n={len(aus):,} expr={expr:.3f} aggr={aggr:.3f}")
    return pd.DataFrame(rows)


def plot_summary_bars(summary: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("expressivity", "Expressivity (mean AU, excl. AU45)"),
        ("aggression", "Aggression mean(AU04/05/07)"),
        ("m_AU25_AU26", "Mouth m = AU25+AU26"),
        ("n_frames", "Successful frames"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    feeds = ["split", "switched"]
    x = np.arange(2)
    width = 0.35
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for i, feed in enumerate(feeds):
            vals = [
                float(summary.loc[(summary.candidate == c) & (summary.feed == feed), col].iloc[0])
                for c in ("Obama", "Romney")
            ]
            ax.bar(x + (i - 0.5) * width, vals, width, label=feed)
        ax.set_xticks(x)
        ax.set_xticklabels(["Obama", "Romney"])
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(
        "2012 debate: split-screen vs switched feed (full tracks, no speaking filter)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_au_heatmap(summary: pd.DataFrame, out: Path) -> None:
    au_cols = [c for c in summary.columns if re.fullmatch(r"AU\d{2}", c)]
    au_cols = sorted(au_cols, key=lambda x: int(x[2:]))
    labels = [f"{r.candidate} · {r.feed}" for r in summary.itertuples()]
    mat = summary[au_cols].copy()
    mat.index = labels
    # order: Obama split, Obama switched, Romney split, Romney switched
    order = [
        "Obama · split",
        "Obama · switched",
        "Romney · split",
        "Romney · switched",
    ]
    mat = mat.loc[order]

    fig, ax = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    sns.heatmap(
        mat,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Mean intensity"},
    )
    ax.set_title("Mean AU intensity — split vs switched (excl. AU45)")
    ax.set_xlabel("Action Unit")
    ax.set_ylabel("")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_aggression_detail(summary: pd.DataFrame, out: Path) -> None:
    long = summary.melt(
        id_vars=["candidate", "feed"],
        value_vars=["AU04", "AU05", "AU07"],
        var_name="au",
        value_name="intensity",
    )
    g = sns.catplot(
        data=long,
        kind="bar",
        x="au",
        y="intensity",
        hue="feed",
        col="candidate",
        height=4.2,
        aspect=0.95,
        palette={"split": "#2563eb", "switched": "#ea580c"},
    )
    g.fig.subplots_adjust(top=0.82)
    g.fig.suptitle(
        "Aggression AUs by feed type (2012, full tracks)",
        fontsize=13,
        fontweight="bold",
    )
    g.set_axis_labels("Action Unit", "Mean intensity")
    g.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(g.fig)


def plot_delta_heatmap(summary: pd.DataFrame, out: Path) -> None:
    """switched − split for each candidate × AU."""
    au_cols = [c for c in summary.columns if re.fullmatch(r"AU\d{2}", c)]
    au_cols = sorted(au_cols, key=lambda x: int(x[2:]))
    deltas = []
    for cand in ("Obama", "Romney"):
        s = summary[(summary.candidate == cand) & (summary.feed == "split")].iloc[0]
        w = summary[(summary.candidate == cand) & (summary.feed == "switched")].iloc[0]
        deltas.append({"candidate": cand, **{au: float(w[au] - s[au]) for au in au_cols}})
    delta = pd.DataFrame(deltas).set_index("candidate")[au_cols]
    vmax = max(0.3, float(np.nanmax(np.abs(delta.values))))
    fig, ax = plt.subplots(figsize=(14, 3.2), constrained_layout=True)
    sns.heatmap(
        delta,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "switched − split"},
    )
    ax.set_title("AU mean difference (switched − split-screen)")
    ax.set_xlabel("Action Unit")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "newExported/comparison/2012_split_vs_switched",
    )
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Prefer newExported split if present and larger, else Exported
    paths = dict(DEFAULT_PATHS)
    for cand in ("Obama", "Romney"):
        neo = ROOT / f"newExported/2012/{cand}_clean_2012/{cand}_clean_2012.csv"
        if neo.is_file():
            paths[(cand, "split")] = neo

    summary = summarize(paths)
    summary.to_csv(out / "summary.csv", index=False)
    plot_summary_bars(summary, out / "summary_bars.png")
    plot_au_heatmap(summary, out / "au_heatmap.png")
    plot_aggression_detail(summary, out / "aggression_bars.png")
    plot_delta_heatmap(summary, out / "au_delta_heatmap.png")
    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
