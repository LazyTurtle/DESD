"""
Scatter plot of Wasserstein distance (X) vs. user-provided values (Y).

Reads the per-dataset Wasserstein distances produced by
wasserstein_log_areas.py (wasserstein_results.csv) and pairs each one, by
dataset name, with a Y value you fill in below.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RESULTS_CSV = Path("wasserstein_results.csv")
OUTPUT_FIGURE = Path("wasserstein_scatter.png")

# Fill in one Y value per dataset name (must match the names used in
# wasserstein_log_areas.py's DATASETS list).
Y_VALUES = {
    "dataset_1": None,
    "dataset_2": None,
    "dataset_3": None,
}


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

    x = [distances[name] for name in common_names]
    y = [Y_VALUES[name] for name in common_names]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=60)
    for name, xi, yi in zip(common_names, x, y):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("Wasserstein distance (log relative area vs. test set)")
    ax.set_ylabel("Y value")

    if len(x) >= 2:
        correlation, p_value = spearmanr(x, y)
        ax.set_title(f"Spearman rho={correlation:.3f}, p={p_value:.4f}")

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=200)
    print(f"Saved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
