"""The results figure.

One style block so the four panels read as one set. Categorical hues assigned
in fixed order and never cycled, one hue light to dark for magnitude, recessive
grid and axes, direct labels where a number is worth reading off, and a legend
whenever two series share a panel.

The figure is a PNG for the README rather than an interactive page, so there is
no hover to fall back on. That is why every panel carries its own labels: the
numbers have to be on the image.
"""
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

SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
SEQUENTIAL = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]

LABELS = {
    "popularity": "Most popular",
    "bert4rec": "BERT4Rec alone",
    "lightgcn": "LightGCN alone",
    "ease": "Recency EASE",
    "two_stage": "Two-stage",
}
ORDER = ["popularity", "bert4rec", "lightgcn", "ease", "two_stage"]


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


def panel_recall(ax, results, k):
    """Recall@k for each system. Bars from zero, so lengths are comparable."""
    values = [results[name][f"recall@{k}"] for name in ORDER]
    colours = [BASELINE] + list(SEQUENTIAL[1:5])
    y = np.arange(len(ORDER))

    ax.barh(y, values, height=0.6, color=colours, zorder=2)
    for yi, value in zip(y, values):
        ax.annotate(f"{value:.3f}", (value, yi), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9, color=INK,
                    fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[name] for name in ORDER])
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_xlabel(f"Recall@{k}, 943 held-out users")
    ax.set_title("Each stage earns its place")
    ax.grid(axis="y", visible=False)
    _strip(ax)


def panel_intervals(ax, comparisons, k):
    """Each system minus the popularity baseline, with its interval.

    The interval is what makes the panel worth drawing: three differences that
    all look like wins are only wins if their intervals clear zero.
    """
    names = ["bert4rec", "lightgcn", "ease", "two_stage"]
    y = np.arange(len(names))

    ax.axvline(0, color=BASELINE, linewidth=1.2, linestyle=(0, (3, 3)),
               zorder=1)
    for yi, name in zip(y, names):
        comparison = comparisons[f"{name}_vs_popularity"]
        ax.hlines(yi, comparison["ci_low"], comparison["ci_high"],
                  color=SERIES_1, linewidth=2.2, alpha=0.45, zorder=2)
        ax.scatter([comparison["difference"]], [yi], s=70, color=SERIES_1,
                   zorder=4, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{comparison['difference']:+.3f}",
                    (comparison["difference"], yi),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=8.5, color=INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[name] for name in names])
    ax.set_ylim(-0.6, len(names) - 0.25)
    ax.set_xlabel(f"Recall@{k} above the popularity baseline")
    ax.set_title("All four clear zero, 2000 paired resamples")
    ax.grid(axis="y", visible=False)
    _strip(ax)


def panel_beta(ax, sweep, k):
    """How much of the final order the reranker should set.

    The panel with a finding in it. At beta = 1 the sequence model reorders the
    candidates on its own opinion and the retrieval score is discarded, which
    is what the competition version did; it is the worst point on the curve,
    below even leaving the retrieval order untouched.
    """
    betas = sorted(float(b) for b in sweep)
    values = [sweep[str(b)] for b in betas]

    ax.plot(betas, values, color=SERIES_1, linewidth=2, zorder=3)
    ax.scatter(betas, values, s=40, color=SERIES_1, zorder=4,
               edgecolor=SURFACE, linewidth=1.6)

    best = max(range(len(betas)), key=lambda i: values[i])
    for i in (0, best, len(betas) - 1):
        ax.annotate(f"{values[i]:.3f}", (betas[i], values[i]),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=8.5, color=INK, fontweight="bold")
    ax.annotate("retrieval order untouched", (betas[0], values[0]),
                textcoords="offset points", xytext=(6, -26), fontsize=7.5,
                color=INK_MUTED)
    ax.annotate("reranker alone, as first written",
                (betas[-1], values[-1]), textcoords="offset points",
                xytext=(-8, 16), ha="right", fontsize=7.5, color=INK_MUTED)

    ax.set_xticks(betas)
    ax.set_xlabel("beta: how much of the order the reranker sets")
    ax.set_ylabel(f"Recall@{k} on validation")
    ax.set_title("Reranking helps only if it keeps the retrieval score")
    ax.grid(axis="x", visible=False)
    _strip(ax)


def panel_coverage(ax, results, k):
    """Recall against catalogue coverage, which is the popularity trap.

    A recommender that returns the same sixty items to everybody scores
    respectably on recall. Plotting the share of the catalogue it reaches is
    what makes that visible.
    """
    # Two-stage and EASE sit almost on top of each other, so the labels
    # alternate above and below rather than collide.
    offsets = {"popularity": (0, 14, "center"), "bert4rec": (0, 14, "center"),
               "lightgcn": (-10, -6, "right"), "ease": (0, -20, "center"),
               "two_stage": (0, 14, "center")}
    for name in ORDER:
        x = results[name]["catalogue_coverage"]
        y = results[name][f"recall@{k}"]
        colour = BASELINE if name == "popularity" else SERIES_1
        ax.scatter([x], [y], s=110, color=colour, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        dx, dy, align = offsets.get(name, (0, 14, "center"))
        ax.annotate(LABELS[name], (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=align, fontsize=8.5,
                    color=INK_SECONDARY)

    ax.set_xlim(0, 0.78)
    ax.set_ylim(-0.012, max(results[n][f"recall@{k}"] for n in ORDER) * 1.4)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of the catalogue ever recommended")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title("Popularity reaches 5% of the catalogue")
    _strip(ax)


def results_figure(summary, path):
    apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    k = summary["k"]

    panel_recall(axes[0, 0], summary["results"], k)
    panel_intervals(axes[0, 1], summary["comparisons"], k)
    panel_beta(axes[1, 0], summary["beta_sweep"], k)
    panel_coverage(axes[1, 1], summary["results"], k)

    figure.suptitle(
        "Two-stage recommendation, measured against what is free",
        fontsize=13, fontweight="bold", color=INK, x=0.012, ha="left", y=0.985,
    )
    figure.text(
        0.012, 0.945,
        "MovieLens 100k, leave-one-out: the last item of each user is the "
        "target and everything before it trains. 943 users, 1,682 items. "
        "Every weight fitted on a split one step further back.",
        fontsize=8.5, color=INK_MUTED, ha="left",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.925])
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
