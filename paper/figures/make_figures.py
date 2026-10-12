"""Generate the result figures of the paper from the released CSVs.

Run from the repository root:  uv run python paper/figures/make_figures.py
Writes paper/figures/fig_swaps.pdf, fig_repro.pdf, fig_ladder.pdf (vector, column width).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data" / "results"
OUT = Path(__file__).resolve().parent
H = [3, 7, 15, 30]

# Validated categorical palette (dataviz reference instance, slots 1-3) + inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
    }
)


def tidy(ax, ylabel=None, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(H)))
    ax.set_xticklabels([str(h) for h in H])
    ax.set_xlabel(r"horizon $\tau$ (sessions)")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)


# ------------------------------------------------------------------ Figure: swaps
def fig_swaps() -> None:
    rc = pd.read_csv(R / "result_controls.csv")
    ash = pd.read_csv(R / "audio_shuffle.csv")
    conds = [
        ("real", "real", BLUE, "o"),
        ("within_shuffle", "same-company swap", ORANGE, "s"),
        ("global_shuffle", "random swap", AQUA, "^"),
    ]
    panels = [
        ("text ridge,\ntemporal", "text"),
        ("text ridge,\ndate-ordered company-disjoint", "text_cb"),
        ("WavLM + past vol,\nticker-disjoint", "audio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1), sharey=True)
    for ax, (title, which) in zip(axes, panels, strict=True):
        for cond, label, color, marker in conds:
            if which == "audio":
                q = ash[(ash.split == "ticker_disjoint") & (ash.condition == cond)]
            else:
                split = "temporal" if which == "text" else "combined"
                q = rc[
                    (rc.dataset == "fincall")
                    & (rc.segment == "test")
                    & (rc.target == "dv")
                    & (rc.split == split)
                    & (rc.model == "ridge_text")
                    & (rc.condition == cond)
                ]
            y = [float(q[q.horizon == h].r2_oos.iloc[0]) for h in H]
            ax.plot(
                range(len(H)),
                y,
                color=color,
                marker=marker,
                label=label,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
        ax.axhline(0, color=INK2, linewidth=0.6)
        tidy(ax, title=title)
    axes[0].set_ylabel(r"$R^2_{\mathrm{OOS}}$ vs persistence ($\Delta v$)")
    axes[0].legend(frameon=False, loc="lower left", handlelength=1.6)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig_swaps.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: repro
def fig_repro() -> None:
    spec = importlib.util.spec_from_file_location(
        "mmt", ROOT / "paper" / "tables" / "make_main_tables.py"
    )
    mmt = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mmt)
    rows = []
    kinds = {
        "HTML text (EC)": "text",
        "HTML text$+$audio (EC)": "text",
        "Sawhney ensemble (EC)": "text",
        "SCSS TMLP, text (DEC)": "text",
        "SCSS TSMixer, price (DEC)": "price",
        "KeFVP, text$+$price (MAEC-15)": "price",
        "KeFVP, text$+$price (MAEC-16)": "price",
    }
    labels = {
        "HTML text (EC)": "HTML, text",
        "HTML text$+$audio (EC)": "HTML, text+audio",
        "Sawhney ensemble (EC)": "Sawhney et al., ensemble",
        "SCSS TMLP, text (DEC)": "SCSS TMLP, text",
        "SCSS TSMixer, price (DEC)": "SCSS TSMixer, price",
        "KeFVP, text$+$price (MAEC-15)": "KeFVP, MAEC-15",
        "KeFVP, text$+$price (MAEC-16)": "KeFVP, MAEC-16",
    }
    for name, a, b in mmt.REPRO:
        for h, x, y in zip(H, a, b, strict=True):
            rows.append(
                {"model": labels[name], "kind": kinds[name], "horizon": h, "pct": (y / x - 1) * 100}
            )
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df.model))
    fig, ax = plt.subplots(figsize=(3.15, 2.55))
    markers = dict(zip(H, ["o", "s", "^", "D"], strict=True))
    for i, m in enumerate(order):
        q = df[df.model == m]
        color = ORANGE if q.kind.iloc[0] == "text" else BLUE
        ax.plot([q.pct.min(), q.pct.max()], [i, i], color=color, linewidth=1.0, alpha=0.5)
        for _, r in q.iterrows():
            ax.plot(
                r.pct,
                i,
                marker=markers[r.horizon],
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                linestyle="none",
            )
    ax.axvline(0, color=INK2, linewidth=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("MSE change when companies are held out (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D  # noqa: E402

    handles = [
        Line2D([], [], color=INK2, marker=markers[h], linestyle="none", label=rf"$\tau$={h}")
        for h in H
    ]
    handles += [
        Line2D([], [], color=ORANGE, linewidth=3, label="text model"),
        Line2D([], [], color=BLUE, linewidth=3, label="price model"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.22),
        ncol=3,
        handlelength=1.2,
        columnspacing=0.9,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_repro.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: ladder
def fig_ladder() -> None:
    t4 = pd.read_csv(R / "result_table_4.csv")
    t1 = pd.read_csv(R / "result_table_1.csv")
    t2 = pd.read_csv(R / "result_table_2.csv")
    stages = [
        ("S0_HAR", "HAR-RV (Stage 0)", INK, "-", "o"),
        ("S1_GBDT", "GBDT + ticker FE (Stage 1)", INK2, "--", "v"),
        ("S2_text", "text ridge + vol (Stage 2)", BLUE, "-", "s"),
        ("S3_audio", "audio ridge + vol (Stage 3)", ORANGE, "-", "^"),
        ("S4_fusion", "stack fusion (Stage 4)", AQUA, "-", "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, (split, title) in zip(
        axes,
        [("temporal", "temporal split"), ("combined", "date-ordered company-disjoint split")],
        strict=True,
    ):
        q4 = t4[
            (t4.dataset == "fincall")
            & (t4.segment == "test")
            & (t4.target == "dv")
            & (t4.split == split)
        ]
        q1 = t1[
            (t1.dataset == "fincall")
            & (t1.convention == "trading")
            & (t1.segment == "test")
            & (t1.target == "dv")
            & (t1.split == split)
        ]
        tm = [float(q1[(q1.model == "train_mean") & (q1.horizon == h)].r2_oos.iloc[0]) for h in H]
        q2 = t2[
            (t2.dataset == "fincall")
            & (t2.segment == "test")
            & (t2.target == "dv")
            & (t2.split == split)
            & (t2.model == "ridge_pastvol")
        ]
        pv = [float(q2[q2.horizon == h].r2_oos.iloc[0]) for h in H]
        ax.plot(
            range(len(H)), pv, color=INK2, linestyle=":", marker="x", label="past-vol ridge S0$'$"
        )
        ax.fill_between(
            range(len(H)), 0, tm, color=GRID, alpha=0.6, linewidth=0, label="train-mean constant"
        )
        for stage, label, color, ls, marker in stages:
            y = [float(q4[(q4.stage == stage) & (q4.horizon == h)].r2_oos.iloc[0]) for h in H]
            ax.plot(
                range(len(H)),
                y,
                color=color,
                linestyle=ls,
                marker=marker,
                label=label,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
        ax.axhline(0, color=INK2, linewidth=0.6)
        tidy(ax, title=title)
        ax.set_ylim(-1.0, 0.65)
    axes[0].set_ylabel(r"$R^2_{\mathrm{OOS}}$ vs persistence ($\Delta v$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        handlelength=1.8,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(w_pad=1.0, rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "fig_ladder.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_swaps()
    fig_repro()
    fig_ladder()
    print("wrote fig_swaps.pdf, fig_repro.pdf, fig_ladder.pdf")
