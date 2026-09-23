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
    "two_stage": "Two-stage",
    "lightgcn": "LightGCN alone",
}
ORDER = ["popularity", "bert4rec", "two_stage", "lightgcn"]


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
    colours = [BASELINE] + list(SEQUENTIAL[1:4])
    y = np.arange(len(ORDER))

    ax.barh(y, values, height=0.6, color=colours, zorder=2)
    for yi, value in zip(y, values):
        ax.annotate(f"{value:.3f}", (value, yi), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9, color=INK,
                    fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[name] for name in ORDER])
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_xlabel(f"Recall@{k}, 940 held-out users")
    ax.set_title("Retrieval alone is the best single system")
    ax.grid(axis="y", visible=False)
    _strip(ax)


def panel_intervals(ax, comparisons, k):
    """Each system minus the popularity baseline, with its interval.

    The interval is what makes the panel worth drawing: three differences that
    all look like wins are only wins if their intervals clear zero.
    """
    names = ["bert4rec", "two_stage", "lightgcn"]
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
    ax.set_title("All three clear zero, 2000 paired resamples")
    ax.grid(axis="y", visible=False)
    _strip(ax)


def panel_alpha(ax, sweep, k):
    """The retrieval blend sweep: how much the content half is worth."""
    alphas = sorted(float(a) for a in sweep)
    values = [sweep[str(a)] if str(a) in sweep else sweep[f"{a}"]
              for a in alphas]

    ax.plot(alphas, values, color=SERIES_1, linewidth=2, zorder=3)
    ax.scatter(alphas, values, s=40, color=SERIES_1, zorder=4,
               edgecolor=SURFACE, linewidth=1.6)
    for alpha, value in zip(alphas, values):
        if alpha in (0.0, 1.0):
            ax.annotate(f"{value:.2f}", (alpha, value),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.5, color=INK,
                        fontweight="bold")

    ax.set_xticks(alphas)
    ax.set_xlabel("alpha: 0 is content only, 1 is the graph only")
    ax.set_ylabel("Target inside the candidate set")
    ax.set_title("Content adds nothing to retrieval here")
    ax.grid(axis="x", visible=False)
    _strip(ax)


def panel_coverage(ax, results, k):
    """Recall against catalogue coverage, which is the popularity trap.

    A recommender that returns the same sixty items to everybody scores
    respectably on recall. Plotting the share of the catalogue it reaches is
    what makes that visible.
    """
    for name in ORDER:
        x = results[name]["catalogue_coverage"]
        y = results[name][f"recall@{k}"]
        colour = BASELINE if name == "popularity" else SERIES_1
        ax.scatter([x], [y], s=110, color=colour, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(LABELS[name], (x, y), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=8.5,
                    color=INK_SECONDARY)

    ax.set_xlim(0, 0.78)
    ax.set_ylim(0, max(results[n][f"recall@{k}"] for n in ORDER) * 1.35)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of the catalogue ever recommended")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title("Popularity reaches 4% of the catalogue")
    _strip(ax)


def results_figure(summary, path):
    apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    k = summary["k"]

    panel_recall(axes[0, 0], summary["results"], k)
    panel_intervals(axes[0, 1], summary["comparisons"], k)
    panel_alpha(axes[1, 0], summary["alpha_sweep"], k)
    panel_coverage(axes[1, 1], summary["results"], k)

    figure.suptitle(
        "Two-stage recommendation, measured against what is free",
        fontsize=13, fontweight="bold", color=INK, x=0.012, ha="left", y=0.985,
    )
    figure.text(
        0.012, 0.945,
        "MovieLens 100k, leave-one-out: the last item of each user is the "
        "target and everything before it trains. 940 users, 1,682 items.",
        fontsize=8.5, color=INK_MUTED, ha="left",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.925])
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
