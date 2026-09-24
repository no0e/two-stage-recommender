"""The results figure.

One style block so the two panels read as one set. Categorical hues assigned
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

SERIES_1 = "#2a78d6"
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


def panel_coverage(ax, results, k):
    """Recall against catalogue coverage, which is the popularity trap.

    A recommender that returns the same sixty items to everybody scores
    respectably on recall. Plotting the share of the catalogue it reaches is
    what makes that visible.
    """
    # Two-stage and EASE sit almost on top of each other, so the labels
    # alternate above and below rather than collide.
    offsets = {"popularity": (12, -3, "left"), "bert4rec": (-12, -3, "right"),
               "lightgcn": (0, -18, "center"), "ease": (13, -4, "left"),
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
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    k = summary["k"]

    panel_recall(axes[0], summary["results"], k)
    panel_coverage(axes[1], summary["results"], k)

    figure.suptitle(
        "Two-stage recommendation, measured against what is free",
        fontsize=13, fontweight="bold", color=INK, x=0.012, ha="left", y=0.99,
    )
    figure.text(
        0.012, 0.912,
        "MovieLens 100k, leave-one-out: the last film of each user is the "
        "target and everything before it trains. 943 users, 1,682 films.",
        fontsize=8.5, color=INK_MUTED, ha="left",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.9])
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
