import argparse

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def get_fold_models(models_folder: Path, only_best: bool = True) -> list[Path]:
    pattern = "**/best.pt" if only_best else "**/*.pt"
    return sorted(models_folder.glob(pattern, recurse_symlinks=True))


def get_test_images(dataset_root: Path) -> list[Path]:
    images_dir = dataset_root / 'images'
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def load_yolo_labels(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO-format label file, returning (classes, boxes_xywhn)."""
    if not label_path.exists():
        return np.zeros((0,)), np.zeros((0, 4))

    classes, boxes = [], []
    with open(label_path) as label_file:
        for line in label_file:
            parts = line.split()
            if not parts:
                continue
            classes.append(float(parts[0]))
            boxes.append([float(v) for v in parts[1:5]])

    if not classes:
        return np.zeros((0,)), np.zeros((0, 4))
    return np.array(classes), np.array(boxes)


def xywhn_to_xyxy(boxes_xywhn: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes_xywhn.size == 0:
        return np.zeros((0, 4))
    cx, cy, w, h = boxes_xywhn[:, 0], boxes_xywhn[:, 1], boxes_xywhn[:, 2], boxes_xywhn[:, 3]
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return np.stack([x1, y1, x2, y2], axis=1)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes, shape (len(boxes1), len(boxes2))."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area1[:, None] + area2[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def match_detections(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
    single_class: bool,
) -> tuple[int, int, int]:
    """Greedily match predictions (highest confidence first) to ground truth boxes.

    Returns (true_positives, false_positives, false_negatives) for one image.
    """
    order = np.argsort(-pred_scores)
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    gt_taken = np.zeros(len(gt_boxes), dtype=bool)
    tp = fp = 0

    for pred_idx in order:
        candidates = np.ones(len(gt_boxes), dtype=bool) if single_class \
            else (gt_classes == pred_classes[pred_idx])
        candidates &= ~gt_taken

        if not candidates.any():
            fp += 1
            continue

        ious = np.where(candidates, iou_matrix[pred_idx], -1.0)
        best_gt = np.argmax(ious)
        if ious[best_gt] >= iou_threshold:
            tp += 1
            gt_taken[best_gt] = True
        else:
            fp += 1

    fn = int((~gt_taken).sum())
    return tp, fp, fn


def evaluate_fold_per_image(
    model_path: Path,
    image_paths: list[Path],
    labels_dir: Path,
    confidence: float,
    iou_match: float,
    iou_nms: float,
    image_size: int,
    single_class: bool,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one model over the whole test set once and return per-image (tp, fp, fn) arrays."""
    model = YOLO(model_path)
    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=confidence,
        iou=iou_nms,
        imgsz=image_size,
        device=device,
        verbose=False,
    )

    tp = np.zeros(len(image_paths), dtype=int)
    fp = np.zeros(len(image_paths), dtype=int)
    fn = np.zeros(len(image_paths), dtype=int)

    for i, (image_path, result) in enumerate(zip(image_paths, results)):
        height, width = result.orig_shape # type: ignore
        gt_classes, gt_boxes_xywhn = load_yolo_labels(labels_dir / f'{image_path.stem}.txt')
        gt_boxes = xywhn_to_xyxy(gt_boxes_xywhn, width, height)

        pred_boxes = result.boxes.xyxy.cpu().numpy() # type: ignore
        pred_scores = result.boxes.conf.cpu().numpy() # type: ignore
        pred_classes = result.boxes.cls.cpu().numpy() # type: ignore

        tp[i], fp[i], fn[i] = match_detections(
            pred_boxes, pred_scores, pred_classes,
            gt_boxes, gt_classes, iou_match, single_class,
        )

    return tp, fp, fn


def f1_from_counts(tp, fp, fn) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def bootstrap_f1_difference(
    folder_a: str | Path,
    folder_b: str | Path,
    dataset_root: str | Path,
    confidence: float,
    iou_match: float = 0.5,
    iou_nms: float = 0.7,
    n_bootstrap: int = 10000,
    image_size: int = 640,
    single_class: bool = False,
    only_best: bool = True,
    device: str | None = None,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict:
    """Bootstrap the F1-score difference between two sets of k-fold models on a shared test set.

    For each bootstrap iteration, a fold model is drawn (with replacement) from `folder_a` and
    another from `folder_b`, the test set is resampled with replacement, and the F1-score
    difference (A - B) at `confidence` is recorded. Returns the observed F1 scores/difference
    together with a percentile confidence interval for the difference.
    """
    dataset_root = Path(dataset_root)
    labels_dir = dataset_root / 'labels'
    image_paths = get_test_images(dataset_root)
    if not image_paths:
        raise ValueError(f'No test images found under {dataset_root / "images"}')

    models_a = get_fold_models(Path(folder_a), only_best=only_best)
    models_b = get_fold_models(Path(folder_b), only_best=only_best)
    if not models_a:
        raise ValueError(f'No fold models found under {folder_a}')
    if not models_b:
        raise ValueError(f'No fold models found under {folder_b}')

    print(f'Found {len(models_a)} fold model(s) for A, {len(models_b)} for B, '
          f'{len(image_paths)} test image(s).')

    eval_kwargs = dict(
        labels_dir=labels_dir, confidence=confidence, iou_match=iou_match, iou_nms=iou_nms,
        image_size=image_size, single_class=single_class, device=device,
    )

    print('Running inference for dataset A folds...')
    counts_a = list()
    for m in tqdm(models_a):
        counts_a.append(evaluate_fold_per_image(m, image_paths, **eval_kwargs)) # type: ignore

    print('Running inference for dataset B folds...')
    counts_b = list()
    for m in tqdm(models_b):
        counts_b.append(evaluate_fold_per_image(m, image_paths, **eval_kwargs)) # type: ignore

    n_images = len(image_paths)
    rng = np.random.default_rng(seed)

    observed_f1_a = float(np.mean([f1_from_counts(*[c.sum() for c in fold]) for fold in counts_a]))
    observed_f1_b = float(np.mean([f1_from_counts(*[c.sum() for c in fold]) for fold in counts_b]))

    diffs = np.empty(n_bootstrap)
    for i in tqdm(range(n_bootstrap), 'Bootstrapping'):
        fold_a = counts_a[rng.integers(len(models_a))]
        fold_b = counts_b[rng.integers(len(models_b))]
        sample_idx = rng.integers(0, n_images, size=n_images)

        f1_a = f1_from_counts(*(c[sample_idx].sum() for c in fold_a))
        f1_b = f1_from_counts(*(c[sample_idx].sum() for c in fold_b))
        diffs[i] = f1_a - f1_b

    alpha = 1 - confidence_level
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        'confidence_threshold': confidence,
        'n_images': n_images,
        'n_folds_a': len(models_a),
        'n_folds_b': len(models_b),
        'n_bootstrap': n_bootstrap,
        'observed_f1_a': observed_f1_a,
        'observed_f1_b': observed_f1_b,
        'observed_diff': observed_f1_a - observed_f1_b,
        'ci_level': confidence_level,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'bootstrap_diffs': diffs,
    }


def _print_report(results: dict) -> None:
    print()
    print(f'F1 @ conf={results["confidence_threshold"]}  '
          f'({results["n_images"]} images, {results["n_bootstrap"]} bootstrap iterations)')
    print(f'  Dataset A F1: {results["observed_f1_a"]:.4f}  ({results["n_folds_a"]} folds)')
    print(f'  Dataset B F1: {results["observed_f1_b"]:.4f}  ({results["n_folds_b"]} folds)')
    print(f'  Difference (A - B): {results["observed_diff"]:.4f}')
    ci_pct = int(results['ci_level'] * 100)
    print(f'  {ci_pct}% bootstrap CI: [{results["ci_low"]:.4f}, {results["ci_high"]:.4f}]')
    significant = results['ci_low'] > 0 or results['ci_high'] < 0
    print(f'  CI excludes zero: {significant}')


def _save_output(results: dict, output_folder: Path) -> None:
    import csv, json

    output_folder.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in results.items() if k != 'bootstrap_diffs'}
    with open(output_folder / 'summary.json', 'w') as summary_file:
        json.dump(summary, summary_file, indent=2)

    with open(output_folder / 'bootstrap_diffs.csv', 'w', newline='') as diffs_file:
        writer = csv.writer(diffs_file, lineterminator='\n')
        writer.writerow(['diff_f1_a_minus_b'])
        for diff in results['bootstrap_diffs']:
            writer.writerow([diff])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bootstrap the 95% CI of the F1-score difference between two sets of '
                     'k-fold YOLO models, evaluated on a shared test set.'
    )
    parser.add_argument('folder_a', type=Path, help='Folder containing the fold weights for dataset A.')
    parser.add_argument('folder_b', type=Path, help='Folder containing the fold weights for dataset B.')
    parser.add_argument('test_data', type=Path, help='Root of the YOLO-format test set (contains images/ and labels/).')
    parser.add_argument('--conf', required=True, type=float, help='Confidence threshold at which to compute F1.')
    parser.add_argument('--iou-match', type=float, default=0.5, help='IoU threshold to count a detection as a match.')
    parser.add_argument('--iou-nms', type=float, default=0.7, help='IoU threshold used for NMS during inference.')
    parser.add_argument('--n-bootstrap', type=int, default=10000, help='Number of bootstrap iterations.')
    parser.add_argument('--image-size', type=int, default=640, help='Inference image size.')
    parser.add_argument('--single-class', action=argparse.BooleanOptionalAction, default=False, help='Ignore class labels when matching detections.')
    parser.add_argument('--only-best', action=argparse.BooleanOptionalAction, default=True, help='Only use "best.pt" weights when scanning the fold folders.')
    parser.add_argument('--device', default=None, help='Inference device, e.g. "cpu" or "0".')
    parser.add_argument('--ci-level', type=float, default=0.95, help='Confidence level for the bootstrap interval.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--output', type=Path, default=None, help='If provided, save a summary.json and bootstrap_diffs.csv here.')

    args = parser.parse_args()

    results = bootstrap_f1_difference(
        folder_a=args.folder_a,
        folder_b=args.folder_b,
        dataset_root=args.test_data,
        confidence=args.conf,
        iou_match=args.iou_match,
        iou_nms=args.iou_nms,
        n_bootstrap=args.n_bootstrap,
        image_size=args.image_size,
        single_class=args.single_class,
        only_best=args.only_best,
        device=args.device,
        confidence_level=args.ci_level,
        seed=args.seed,
    )

    _print_report(results)

    if args.output is not None:
        _save_output(results, args.output)
        print(f'\nSaved results to {args.output}')
