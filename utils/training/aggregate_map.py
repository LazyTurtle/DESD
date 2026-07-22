import csv, argparse
from statistics import mean, stdev
from pathlib import Path

MAP_GLOB_PATTERN = "**/map.csv"
OUTPUT_MEAN = "mAP_mean.csv"
OUTPUT_STDS = "mAP_std.csv"

def aggregate_map_kfold(folds_root:str|Path, output_folder:str|Path|None=None, target_filename:str|None=None):
    folds_root = Path(folds_root)
    output_folder = Path(output_folder) if output_folder is not None else folds_root
    output_folder.mkdir(parents=True, exist_ok=True)

    mAP_by_confidence = {}

    pattern = f"**/{target_filename}" if target_filename is not None else MAP_GLOB_PATTERN
    files = folds_root.glob(pattern)

    for file in files:

        with open(file, mode="r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for row in reader:
                if len(row) >= 2:
                    conf_label = row[0].strip()
                    map_val = float(row[1].strip())

                    if conf_label not in mAP_by_confidence:
                        mAP_by_confidence[conf_label] = []
                    mAP_by_confidence[conf_label].append(map_val)

    mean_rows = []
    std_rows = []

    for conf_label, map_values in mAP_by_confidence.items():
        avg_map = mean(map_values)
        std_map = stdev(map_values) if len(map_values) > 1 else 0.0

        mean_rows.append([conf_label, avg_map])
        std_rows.append([conf_label, std_map])

    with open(output_folder / OUTPUT_MEAN, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Confidence", "Mean_mAP"])
        writer.writerows(mean_rows)

    with open(output_folder / OUTPUT_STDS, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Confidence", "Std_mAP"])
        writer.writerows(std_rows)

    print("Aggregation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate training results from all folds.")
    parser.add_argument('folds_folder', type=Path, help='Root folder for all folds.')
    parser.add_argument('--output', type=Path, default=None, required=False, help='Output folder for the aggregated csv files. Default [folds_folder]')
    parser.add_argument('--target', type=str, default=None, required=False, help='Target namefile to aggregate. If not provided all csv files will be collected.')
    args = parser.parse_args()

    aggregate_map_kfold(args.folds_folder, args.output, args.target)
