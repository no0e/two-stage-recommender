"""The results figure: both catalogues, and the weight that was fitted twice.

Three panels. The first two are the same four systems on the two Amazon
categories, drawn on their own axes because the problems are not the same
difficulty: a hit at rank ten out of 25,612 items and a hit out of 162,035 are
different events, and putting them on a shared axis would only say that the
larger catalogue is harder, which is not news.

The third is the panel with a finding in it. beta is how much of the final
order the sequence model sets, fitted separately on each category against
models that never saw a test target. It came out at 0.75 both times, and both
curves fall again at 1.0: the reranker is worth most of the order and not all
of it.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SEQUENTIAL = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]

LABELS = {
    "popularity": "Most popular",
    "bert4rec": "BERT4Rec alone",
    "lightgcn": "LightGCN alone",
    "two_stage": "Two-stage",
}
ORDER = ["popularity", "bert4rec", "lightgcn", "two_stage"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def _strip(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def panel_recall(ax, summary, k):
    """Recall@k for each system. Bars from zero, so lengths are comparable."""
    results = summary["results"]
    values = [results[name][f"recall@{k}"] for name in ORDER]
    colours = [BASELINE, SEQUENTIAL[1], SEQUENTIAL[1], SERIES_1]
    y = np.arange(len(ORDER))

    ax.barh(y, values, height=0.6, color=colours, zorder=2)
    for yi, value in zip(y, values):
        ax.annotate(f"{value:.4f}", (value, yi), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9, color=INK,
                    fontweight="bold")

    dataset = summary["dataset"]
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[name] for name in ORDER])
    ax.set_xlim(0, max(values) * 1.32)
    ax.set_xlabel(f"Recall@{k}, {dataset['items_in_catalogue']:,} items")
    ax.set_title(dataset["category"].replace("_", " "))
    ax.grid(axis="y", visible=False)
    _strip(ax)


def panel_beta(ax, summaries, k):
    """How much of the final order the reranker should set, on each category."""
    colours = (SERIES_1, SERIES_2)
    for summary, colour in zip(summaries, colours):
        sweep = summary["beta_sweep"]
        betas = sorted(float(b) for b in sweep)
        # Each category is drawn against its own best, so two problems of very
        # different difficulty can share one axis and still be compared.
        values = [sweep[str(b)] for b in betas]
        best = max(values)
        share = [v / best for v in values]

        label = summary["dataset"]["category"].replace("_", " ")
        ax.plot(betas, share, color=colour, linewidth=2, zorder=3, label=label)
        ax.scatter(betas, share, s=34, color=colour, zorder=4,
                   edgecolor=SURFACE, linewidth=1.5)

    ax.axvline(0.75, color=BASELINE, linewidth=1.2, linestyle=(0, (3, 3)),
               zorder=1)
    ax.annotate("both fitted to 0.75", (0.75, 1.035),
                ha="center", fontsize=8.5, color=INK, fontweight="bold")

    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0.55, 1.09)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("beta: how much of the order the reranker sets")
    ax.set_ylabel(f"Recall@{k}, against each category's best")
    ax.set_title("Reranking helps only if it keeps the retrieval score")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    _strip(ax)


def results_figure(summaries, path):
    """`summaries` is the list of run summaries, smaller catalogue first."""
    apply_style()
    summaries = sorted(summaries,
                       key=lambda s: s["dataset"]["items_in_catalogue"])
    k = summaries[0]["k"]

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    panel_recall(axes[0], summaries[0], k)
    panel_recall(axes[1], summaries[1], k)
    panel_beta(axes[2], summaries, k)

    figure.suptitle(
        "LightGCN retrieval, BERT4Rec reranking, on two Amazon catalogues",
        fontsize=13, fontweight="bold", color=INK, x=0.008, ha="left", y=0.98)
    figure.text(
        0.008, 0.915,
        "Amazon Reviews 2023, 5-core, leave-one-out: the last item of each "
        "user is the target and everything before it trains. 20,000 held-out "
        "users per category, every weight fitted on a split one step further "
        "back.",
        fontsize=8.5, color=INK_MUTED, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.88])
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_from_docs(docs_dir, path):
    """Draw from whatever `amazon_*.json` files a run has left behind."""
    summaries = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(Path(docs_dir).glob("amazon_*.json"))]
    if len(summaries) < 2:
        return None
    return results_figure(summaries, path)
