import os, csv, argparse
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = r"D:\workspace\yolo_training\data\fold_evaluation"
# colour palette (colorblind safe, Paul Tol's Bright)
PALETTE = [
    '#4477AA',
    '#EE6677',
    '#228833',
    '#CCBB44',
    '#66CCEE',
    '#AA3377',
    '#BBBBBB',
]

LINE_STYLES = ['solid', "dotted", (0, (3, 1, 1, 1, 1, 1)), "dashed", 'dashdot']

def load_csv_data(filepath):
    data = {}
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            for col in reader.fieldnames:
                data[col] = []

            for row in reader:
                for col, val in row.items():
                    try:
                        value = float(val)
                    except:
                        value = None
                    
                    data[col].append(value)
    return data


def process_folder(root_dir:str|Path):
    models_data = {}  # Format: {metric_name: {model_name: {'means': [...], 'stds': [...]}}}

    for entry in os.scandir(root_dir):
        if entry.is_dir():
            model_name = entry.name
            csv_files = [f for f in os.listdir(entry.path) if f.endswith(".csv")]

            if len(csv_files) < 2:
                print(
                    f"Skipping '{model_name}': expected 2 CSV files, found {len(csv_files)}."
                )
                continue

            # find mean vs std file by filename keyword, or fall back to alphabetical order
            mean_file = next((f for f in csv_files if "mean" in f.lower()), None)
            std_file = next((f for f in csv_files if "std" in f.lower()), None)

            if not mean_file or not std_file:
                csv_files.sort()
                mean_file, std_file = csv_files[0], csv_files[1]

            means = load_csv_data(os.path.join(entry.path, mean_file))
            stds = load_csv_data(os.path.join(entry.path, std_file))

            for metric in means:
                if metric not in models_data:
                    models_data[metric] = {}

                models_data[metric][model_name] = {
                    "means": means[metric],
                    "stds": stds.get(metric, [0.0] * len(means[metric])),
                }

    return models_data


def plot_metrics(models_data, output_folder:str|Path):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for metric, models in models_data.items():

        # plot 1: mean values only
        plt.figure(figsize=(10, 6))
        for index, (model_name, values) in enumerate(models.items()):
            x_steps = list(range(1, len(values["means"]) + 1))
            plt.plot(
                x_steps, 
                values["means"], 
                label=model_name, 
                linewidth=2,
                color=PALETTE[index % len(PALETTE)],
                linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            )

        title = f"{metric.upper()} - Means Across Models"
        title = title.replace('/','_')
        title = title.replace('\\','_')
        plt.title(title, fontsize=14)
        plt.xlabel("Step / Epoch", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(title="Models")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        save_path = output_folder.joinpath(title+'.png')
        plt.savefig(save_path, dpi=800, bbox_inches="tight")
        plt.close()

        # plot 2: means with standard deviation Area
        plt.figure(figsize=(10, 6))
        for index, (model_name, values) in enumerate(models.items()):
            means = values["means"]
            stds = values["stds"]
            x_steps = list(range(1, len(means) + 1))

            upper_bound = [m + s  if m is not None and s is not None else 0.0 for m, s in zip(means, stds)]
            lower_bound = [m - s  if m is not None and s is not None else 0.0 for m, s in zip(means, stds)]

            (line,) = plt.plot(
                x_steps, 
                means, 
                label=f"{model_name}", 
                linewidth=2,
                color=PALETTE[index % len(PALETTE)],
                linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            )
            plt.fill_between(
                x_steps,
                lower_bound,
                upper_bound,
                color=line.get_color(),
                alpha=0.2,
            )
        title = f"{metric.upper()} - Means with Standard Deviation Shading"
        title = title.replace('/','_')
        title = title.replace('\\','_')
        plt.title(
            title,
            fontsize=14,
        )
        plt.xlabel("Step / Epoch", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(title="Models")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        save_path = output_folder.joinpath(title+'.png')
        plt.savefig(save_path, dpi=800, bbox_inches="tight")
        plt.close()

def plot(root_folder:str|Path, output_folder:str|Path):
    data = process_folder(root_folder)
    plot_metrics(data, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CSV files representing  K-fold Cross Validation training and evaluation data.")
    parser.add_argument('root', type=Path, help='Root folder for all CSV files.')
    parser.add_argument('--output', type=Path, default=None, required=False, help='Output folder for the plot files. Default [root]')
    args = parser.parse_args()
    out = args.output if args.output is not None else args.root
    plot(args.root, out)