"""
Scatter plot of Wasserstein distance (X) vs. user-provided values (Y).

Reads the per-dataset Wasserstein distances produced by
wasserstein_log_areas.py (wasserstein_results.csv) and pairs each one, by
dataset name, with a Y value you fill in below.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mk
from scipy.stats import bootstrap, spearmanr

PALETTE = [
    '#4477AA',
    '#EE6677',
    '#228833',
    '#CCBB44',
    '#66CCEE',
    '#AA3377',
    '#BBBBBB',
    '#4477AA',
    '#EE6677',
    '#228833',
    '#444444',
    '#444444',
    '#444444',
    '#444444',
    '#444444',
    '#444444',
]

MARKERS = [
    "o",
    "o",
    "o",
    "o",
    "o",
    "o",
    "o",
    "o",
    "o",
    "o",
    "p",
    "p",
    "p",
    "p",
    "p",
    "p",
]


RESULTS_CSV = Path("logs/wasserstein/wasserstein_results.csv")
OUTPUT_FIGURE = Path("logs/wasserstein/wasserstein_scatter.png")

# Fill in one Y value per dataset name (must match the names used in
# wasserstein_log_areas.py's DATASETS list).
Y_VALUES = {
    'ACFR' : 0.235,
    'Agroscope' : 0.294,
    'APPLE MOTS' : 0.575,
    'Deep Fruits' : 0.356,
    'Kfuji' : 0.403,
    'MetaFruit' : 0.614,
    'MinneApple' : 0.630,
    'Open Access RGBD' : 0.234,
    'SMA' : 0.338,
    'WSU' : 0.263,
    'Synthetic A' : 0.378,
    'Synthetic B' : 0.453,
    'Synthetic C' : 0.470,
    'SA4' : 0.456,
    'SB4' : 0.501,
    'SC3' : 0.521,
}

# Bootstrap CI for Spearman's rho. (X, Y) pairs are resampled together
# (with replacement) each iteration, rho is recomputed, and the CI is the
# percentile interval of the resulting distribution. Meaningless below ~3-4
# datasets - there just aren't enough points for resampling to say anything.
N_BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 42
MIN_DATASETS_FOR_BOOTSTRAP = 4
BOOTSTRAP_BATCH_SIZE = 1000


def load_distances(results_csv: Path) -> dict:
    if not results_csv.is_file():
        raise FileNotFoundError(
            f"{results_csv} not found - run wasserstein_log_areas.py first."
        )
    distances = {}
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            distances[row["dataset"]] = float(row["wasserstein_distance"])
    return distances


def main() -> None:
    distances = load_distances(RESULTS_CSV)

    missing_y = [name for name, y in Y_VALUES.items() if y is None]
    if missing_y:
        raise ValueError(
            f"Y_VALUES is missing values for: {missing_y}. "
            "Fill them in before plotting."
        )

    common_names = [name for name in distances if name in Y_VALUES]
    if not common_names:
        raise ValueError(
            "No dataset names in common between wasserstein_results.csv "
            "and Y_VALUES."
        )
    xs = [distances[name] for name in common_names]
    ys = [Y_VALUES[name] for name in common_names]
    fig, ax = plt.subplots(figsize=(7, 6))
    for x, y, c, m in zip(xs, ys, PALETTE, MARKERS):
        ax.scatter(
            x,
            y,
            s=60,
            c=c,
            marker=m
        )
    for name, xi, yi in zip(common_names, xs, ys):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("Wasserstein distance (log relative area vs. OpenIoT)")
    ax.set_ylabel("Peak F1 Score")

    if len(xs) >= 2:
        correlation, p_value = spearmanr(xs, ys)
        title = f"Spearman rho={correlation:.3f}, p={p_value:.4f}"

        if len(xs) >= MIN_DATASETS_FOR_BOOTSTRAP:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            boot_result = bootstrap(
                (np.asarray(xs), np.asarray(ys)),
                statistic=lambda a, b: spearmanr(a, b).statistic,
                n_resamples=N_BOOTSTRAP_RESAMPLES,
                batch=BOOTSTRAP_BATCH_SIZE,
                paired=True,
                vectorized=False,
                confidence_level=CONFIDENCE_LEVEL,
                method="percentile",
                rng=rng,
            )
            ci_low = boot_result.confidence_interval.low
            ci_high = boot_result.confidence_interval.high
            title += (
                f"\n{CONFIDENCE_LEVEL * 100:.0f}% bootstrap CI="
                f"[{ci_low:.3f}, {ci_high:.3f}] (resamples={N_BOOTSTRAP_RESAMPLES})"
            )
            print(
                f"Spearman rho={correlation:.3f}, {CONFIDENCE_LEVEL * 100:.0f}% "
                f"bootstrap CI=[{ci_low:.3f}, {ci_high:.3f}]"
            )
        else:
            print(
                f"Only {len(xs)} datasets - skipping bootstrap CI "
                f"(need >= {MIN_DATASETS_FOR_BOOTSTRAP})."
            )

        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=800)
    print(f"Saved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
