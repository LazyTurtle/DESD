"""
Compare a series of YOLO datasets against a held-out test dataset by the
distribution of their annotation box areas (relative to image area).

For each dataset:
  1. Collect every bounding-box relative area (w * h, both already normalized
     in [0, 1] by the YOLO format) from its labels/ directory.
  2. Take the natural log of those areas.
  3. Compute the 1D Wasserstein distance (scipy.stats.wasserstein_distance)
     between this dataset's log-area distribution and the test dataset's.

The Spearman correlation is NOT derived from the areas themselves: fill in
SPEARMAN_Y_VALUES below with one value per dataset (e.g. a downstream metric
you already have), and the script will correlate it against the Wasserstein
distances via scipy.stats.spearmanr.

Everything (per-dataset stats and the final correlation) is written to
LOG_FILE. A CSV with the raw distances is also written so the plotting
script (plot_wasserstein_scatter.py) can pick them up without retyping
numbers by hand.
"""

import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, wasserstein_distance

from ..training.my_logging import get_logger
# --------------------------------------------------------------------------
# Configuration - edit this section for your run.
# --------------------------------------------------------------------------

# Held-out test dataset: directory containing YOLO .txt label files
# (one file per image, each line: "class x_center y_center width height").
TEST_LABELS_DIR = Path(r"C:\Users\Emys\Pictures\OpenIoT\labels")

# Series of datasets to compare against the test set: (name, labels_dir).
DATASETS = [
    ('ACFR', Path(r"C:\Users\Emys\Pictures\apples\ACFR\labels")),
    ('Agroscope', Path(r"C:\Users\Emys\Pictures\apples\Agroscope Apple\labels")),
    ('APPLE MOTS', Path(r"C:\Users\Emys\Pictures\apples\APPLE MOTS\labels")),
    ('Deep Fruits', Path(r"C:\Users\Emys\Pictures\apples\Deep Fruits\labels")),
    ('Kfuji', Path(r"C:\Users\Emys\Pictures\apples\Kfuji\labels")),
    ('MetaFruit', Path(r"C:\Users\Emys\Pictures\apples\MetaFruit\labels")),
    ('MinneApple', Path(r"C:\Users\Emys\Pictures\apples\MinneApple\labels")),
    ('Open Access RGBD', Path(r"C:\Users\Emys\Pictures\apples\Open Access RGBD\labels")),
    ('SMA', Path(r"C:\Users\Emys\Pictures\apples\SMA\labels")),
    ('WSU', Path(r"C:\Users\Emys\Pictures\apples\WSU\labels")),
    ('Synthetic A', Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic A\labels")),
    ('Synthetic B', Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic B\labels")),
    ('Synthetic C', Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic C\labels")),
    ('SA4', Path(r"C:\Users\Emys\Pictures\SyntAC\SA\SA4\labels")),
    ('SB4', Path(r"C:\Users\Emys\Pictures\SyntAC\SB\SB4\labels")),
    ('SC3', Path(r"C:\Users\Emys\Pictures\SyntAC\SC\SC3\labels")),
]

# One value per dataset above, in the same order, to correlate (Spearman)
# against the Wasserstein distances. Leave as None to skip the correlation.
SPEARMAN_Y_VALUES = [
    0.235,
    0.294,
    0.575,
    0.356,
    0.403,
    0.614,
    0.630,
    0.234,
    0.338,
    0.263,
    0.378,
    0.453,
    0.470,
    0.456,
    0.501,
    0.521,
]
# Example:
# SPEARMAN_Y_VALUES = [0.812, 0.774, 0.699]

LOG_FILE = Path("logs/wasserstein/wasserstein_analysis.log")
RESULTS_CSV = Path("logs/wasserstein/wasserstein_results.csv")

# Guard against log(0) for degenerate zero-area boxes.
AREA_EPSILON = 1e-12

# --------------------------------------------------------------------------


def setup_logger(log_file: Path):
    logger = get_logger("wasserstein_log_areas", out_folder='logs/wasserstein')
    return logger


def collect_relative_areas(labels_dir: Path) -> np.ndarray:
    """Read every YOLO .txt label file in labels_dir and return an array of
    w * h relative box areas across all files."""
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    areas = []
    for label_file in tqdm(sorted(labels_dir.glob("*.txt")), f'Collecting areas from {labels_dir.parent.stem}'):
        with open(label_file) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(
                        f"Malformed YOLO annotation in {label_file} "
                        f"(line {line_no}): {line!r}"
                    )
                width, height = float(parts[3]), float(parts[4])
                areas.append(width * height)

    if not areas:
        raise ValueError(f"No annotations found under {labels_dir}")

    return np.asarray(areas, dtype=np.float64)


def log_relative_areas(labels_dir: Path) -> np.ndarray:
    areas = collect_relative_areas(labels_dir)
    return np.log(areas + AREA_EPSILON)


def main() -> None:
    logger = setup_logger(LOG_FILE)

    logger.info("Loading test dataset: %s", TEST_LABELS_DIR)
    test_log_areas = log_relative_areas(TEST_LABELS_DIR)
    logger.info(
        "Test dataset: %d annotations, log-area mean=%.4f, std=%.4f",
        test_log_areas.size,
        test_log_areas.mean(),
        test_log_areas.std(),
    )

    names = []
    distances = []

    for name, labels_dir in DATASETS:
        logger.info("Processing dataset '%s': %s", name, labels_dir)
        log_areas = log_relative_areas(labels_dir)

        distance = wasserstein_distance(log_areas, test_log_areas)

        logger.info(
            "Dataset '%s': %d annotations, log-area mean=%.4f, std=%.4f, "
            "wasserstein_distance(vs test)=%.6f",
            name,
            log_areas.size,
            log_areas.mean(),
            log_areas.std(),
            distance,
        )

        names.append(name)
        distances.append(distance)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "wasserstein_distance"])
        for name, distance in zip(names, distances):
            writer.writerow([name, distance])
    logger.info("Wrote Wasserstein distances to %s", RESULTS_CSV)

    if SPEARMAN_Y_VALUES is None:
        logger.info(
            "SPEARMAN_Y_VALUES not set - skipping Spearman correlation."
        )
    else:
        if len(SPEARMAN_Y_VALUES) != len(distances):
            raise ValueError(
                f"SPEARMAN_Y_VALUES has {len(SPEARMAN_Y_VALUES)} entries "
                f"but there are {len(distances)} datasets."
            )
        correlation, p_value = spearmanr(distances, SPEARMAN_Y_VALUES)
        logger.info(
            "Spearman correlation (wasserstein_distance vs provided values): "
            "rho=%.4f, p=%.6f",
            correlation,
            p_value,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
