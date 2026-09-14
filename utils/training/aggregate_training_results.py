import argparse

from pathlib import Path

import csv
from statistics import mean, stdev
from math import isnan

# Configuration
PARENT_FOLDER = r"D:\workspace\yolo_training\data\training\PUBLIC_DATASETS\ACFR"  # Replace with your folder path
OUTPUT_MEAN = "aggregated_mean.csv"
OUTPUT_STD = "aggregated_std.csv"


def aggregate_fold_results(folds_root:str|Path, output_folder:str|Path|None=None, target_filename:str|None=None):
    folds_root = Path(folds_root)
    output_folder = Path(output_folder) if output_folder is not None else folds_root
    # data_grid[(row_idx, col_name)] = [val_from_fold1, val_from_fold2, ...]
    data_grid = {}
    fieldnames_order = []
    max_rows = 0

    pattern = f"**/{target_filename}" if target_filename is not None else "**/*.csv"
    files = folds_root.glob(pattern)

    for file in files:
        if 'map' in file.stem.lower():
            continue
        with open(file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in fieldnames_order:
                        fieldnames_order.append(field)

            for row_idx, row in enumerate(reader):
                max_rows = max(max_rows, row_idx + 1)
                for col_name, val in row.items():
                    if val is not None and val.strip() != "":
                        try:
                            float_val = float(val)
                            if isnan(float_val):
                                print(f'At row {row_idx} I could not convert value {val} of column {col_name} to float, skipping data.')
                                continue
                            key = (row_idx, col_name)
                            data_grid.setdefault(key, []).append(float_val)
                        except ValueError:
                            print(f'At row {row_idx} I could not convert value {val} of column {col_name} to float.')

    if not fieldnames_order:
        print("No valid CSV files found or files were empty.")
        return

    mean_rows = []
    std_rows = []

    for row_idx in range(max_rows):
        mean_row = {}
        std_row = {}

        for col_name in fieldnames_order:
            values = data_grid.get((row_idx, col_name), [])

            if values:
                m = round(mean(values), 6)

                if len(values) > 1:
                    st = round(stdev(values), 6)
                else:
                    st = 0.0

                mean_row[col_name] = m
                std_row[col_name] = st
            else:
                mean_row[col_name] = ""
                std_row[col_name] = ""

        mean_rows.append(mean_row)
        std_rows.append(std_row)

    output_folder.mkdir(parents=True, exist_ok=True)
    for output_file, rows in [(OUTPUT_MEAN, mean_rows), (OUTPUT_STD, std_rows)]:
        with open(output_folder / output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_order)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Successfully generated '{output_folder / OUTPUT_MEAN}' and '{output_folder / OUTPUT_STD}' from {max_rows} max row(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate training results from all folds.")
    parser.add_argument('folds_folder', type=Path, help='Root folder for all folds.')
    parser.add_argument('--output', type=Path, default=None, required=False, help='Output folder for the aggregated csv files. Default [folds_folder]')
    parser.add_argument('--target', type=str, default=None, required=False, help='Target namefile to aggregate. If not provided all csv files will be collected.')
    args = parser.parse_args()

    aggregate_fold_results(args.folds_folder, args.output, args.target)
