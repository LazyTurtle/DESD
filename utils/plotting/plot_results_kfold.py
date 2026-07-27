import os, csv, argparse
import matplotlib.pyplot as plt
from pathlib import Path

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
FIG_SIZE = (16, 9)
X_LABEL = 'Epoch'

TITLE_FONTSIZE = 24
TEXT_FONTSIZE = 20

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
            csv_files = [f for f in os.listdir(entry.path) if f.endswith(".csv") and 'map' not in f.lower()]

            if len(csv_files) < 2:
                print(f"Skipping '{model_name}': expected 2 CSV files, found {len(csv_files)}.")
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

            if 'Confidence' in means.keys():
                models_data['x_values'] = \
                    means["Confidence"] if 'x_values' not in models_data.keys() else \
                        max(models_data['x_values'], means["Confidence"], key=lambda x: len(x))
                
            if 'epoch' in means.keys():
                models_data['x_values'] = \
                    means["epoch"] if 'x_values' not in models_data.keys() else \
                        max(models_data['x_values'], means["epoch"], key=lambda x: len(x))

    return models_data


def plot_metrics(models_data:dict[str,dict], output_folder:str|Path):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    x_values = models_data.pop('x_values') if 'x_values' in models_data.keys() else None
    for metric, models in models_data.items():
        metric = metric.replace('/',' ')
        metric = metric.replace('\\',' ')
        metric = metric.replace('_',' ')
        metric = metric.title()

        # plot 1: mean values only
        fig, ax = plt.subplots()
        ax = plt.figure(figsize=FIG_SIZE)
        for index, (model_name, values) in enumerate(models.items()):
            x_steps = list(x_values) if x_values is not None else list(range(1, len(values["means"]) + 1))
            y_values:list = values['means']
            y_values.extend([None] * max(0, (len(x_steps) - len(y_values))))
            plt.plot(
                x_steps, 
                y_values, 
                label=model_name, 
                linewidth=2,
                color=PALETTE[index % len(PALETTE)],
                linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            )

        title = f"{metric.upper()} - Means Across Datasets"
        plt.title(title, fontsize=TITLE_FONTSIZE)
        plt.xlabel(X_LABEL, fontsize=TEXT_FONTSIZE)
        plt.ylabel(metric, fontsize=TEXT_FONTSIZE)
        plt.yticks(fontsize = TEXT_FONTSIZE)
        plt.xticks(fontsize = TEXT_FONTSIZE)
        ax.legend(
            fontsize=TEXT_FONTSIZE,
            loc='lower center',
            ncol=round(len(models)/2),
            bbox_to_anchor=(0.5, -0.12),
        )
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        save_path = output_folder.joinpath(title+'.png')
        plt.savefig(save_path, dpi=800, bbox_inches="tight")
        plt.close()
        print(f'Saved {save_path}')

        # plot 2: means with standard deviation Area
        fig, ax = plt.subplots()
        ax = plt.figure(figsize=FIG_SIZE)
        for index, (model_name, values) in enumerate(models.items()):
            means = values["means"]
            stds = values["stds"]
            x_steps = list(x_values) if x_values is not None else list(range(1, len(values["means"]) + 1))

            means.extend([None] * max(0, (len(x_steps) - len(means))))
            stds.extend([None] * max(0, (len(x_steps) - len(stds))))

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
        plt.title(title, fontsize=TITLE_FONTSIZE)
        plt.xlabel(X_LABEL, fontsize=TEXT_FONTSIZE)
        plt.ylabel(metric, fontsize=TEXT_FONTSIZE)
        plt.yticks(fontsize = TEXT_FONTSIZE)
        plt.xticks(fontsize = TEXT_FONTSIZE)
        ax.legend(
            fontsize=TEXT_FONTSIZE,
            loc='lower center',
            ncol=round(len(models)/2),
            bbox_to_anchor=(0.5, -0.12),
        )
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        save_path = output_folder.joinpath(title+'.png')
        plt.savefig(save_path, dpi=800, bbox_inches="tight")
        plt.close()
        print(f'Saved {save_path}')

def plot_map(root_folder:str|Path, output_folder:str|Path):
    root_folder = Path(root_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    runs_data = {}

    for entry in os.listdir(root_folder):
        run_path = os.path.join(root_folder, entry)

        if os.path.isdir(run_path):
            run_name = entry
            runs_data[run_name] = {}

            for file_name in os.listdir(run_path):
                if file_name.endswith(".csv") and 'map' in file_name.lower():
                    file_path = os.path.join(run_path, file_name)

                    with open(file_path, mode="r", newline="") as f:
                        reader = list(csv.reader(f))
                        if not reader:
                            continue

                        header = reader[0]
                        is_mean = any("Mean_mAP" in col for col in header)
                        is_std = any("Std_mAP" in col for col in header)

                        for row in reader[1:3]:
                            if len(row) >= 2:
                                conf_label = row[0].strip()
                                val = float(row[1].strip())

                                if conf_label not in runs_data[run_name]:
                                    runs_data[run_name][conf_label] = {}

                                if is_mean:
                                    runs_data[run_name][conf_label]["mean"] = val
                                elif is_std:
                                    runs_data[run_name][conf_label]["std"] = val

    run_names = sorted(list(runs_data.keys()))
    if not run_names:
        print("No run folders found.")
        return

    conf_labels = []
    for r in run_names:
        for c in runs_data[r]:
            if c not in conf_labels:
                conf_labels.append(c)

    num_runs = len(run_names)
    x_indices = list(range(num_runs))

    bar_width = 0.35

    # --- PLOT 1: Mean mAP Values Only ---
    plt.figure(figsize=FIG_SIZE)

    bar_heights50 = [runs_data[run].get('50', {}).get("mean", 0.0) for run in run_names]
    colors = 2*PALETTE[:len(run_names)]
    offset = -0.5 * bar_width
    x_positions = [x + offset for x in x_indices]
    plt.bar(
        x = x_positions,
        height=bar_heights50,
        width=bar_width,
        color=colors,
        label="mAP50",
        edgecolor = 'Black',
        alpha=0.8,
    )
    offset = 0.5 * bar_width
    x_positions = [x + offset for x in x_indices]
    bar_heights95 = [runs_data[run].get('95', {}).get("mean", 0.0) for run in run_names]

    plt.bar(
        x=x_positions,
        height=bar_heights95,
        width=bar_width,
        color=colors,
        hatch = '///',
        label="mAP50-95",
        edgecolor = 'Black',
        alpha=0.8,
    )

    # plt.xlabel("Datasets")
    plt.ylabel("Mean mAP", fontsize=TEXT_FONTSIZE)
    title = "mAP Values per Run by Confidence Level"
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.yticks(fontsize = TEXT_FONTSIZE)
    plt.xticks(x_indices, run_names, ha="right", fontsize=TEXT_FONTSIZE)
    plt.legend(fontsize=TEXT_FONTSIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_path = output_folder.joinpath(title+'.png')
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    plt.close()
    print(f'Saved {save_path}')

    # --- PLOT 2: Mean mAP Values with Standard Deviation Error Bars ---
    plt.figure(figsize=FIG_SIZE)

    bar_heights50 = [runs_data[run].get('50', {}).get("mean", 0.0) for run in run_names]
    stds50 = [runs_data[r].get('50', {}).get("std", 0.0) for r in run_names]
    colors = 2*PALETTE[:len(run_names)]
    offset = -0.5 * bar_width
    x_positions = [x + offset for x in x_indices]
    plt.bar(
        x = x_positions,
        height=bar_heights50,
        width=bar_width,
        color=colors,
        label="mAP50",
        edgecolor = 'Black',
        yerr=stds50,
        error_kw={"ecolor": "black", "linewidth": 1.5},
        alpha=0.8,
    )

    for x_pos, mean, std in zip(x_positions, bar_heights50, stds50):
        plt.fill_between(
            [x_pos - bar_width / 2, x_pos + bar_width / 2],
            [mean - std, mean - std],
            [mean + std, mean + std],
            color="black",
            alpha=0.15,
        )

    bar_heights95 = [runs_data[run].get('95', {}).get("mean", 0.0) for run in run_names]
    stds95 = [runs_data[r].get('95', {}).get("std", 0.0) for r in run_names]
    offset = 0.5 * bar_width
    x_positions = [x + offset for x in x_indices]

    plt.bar(
        x=x_positions,
        height=bar_heights95,
        width=bar_width,
        color=colors,
        hatch = '///',
        label="mAP50-95",
        edgecolor = 'Black',
        yerr=stds95,
        error_kw={"ecolor": "black", "linewidth": 1.5},
        alpha=0.8,
    )
    for x_pos, mean, std in zip(x_positions, bar_heights95, stds95):
        plt.fill_between(
            [x_pos - bar_width / 2, x_pos + bar_width / 2],
            [mean - std, mean - std],
            [mean + std, mean + std],
            color="black",
            alpha=0.15,
        )


    # plt.xlabel("Datasets")
    plt.ylabel("Mean mAP", fontsize=TEXT_FONTSIZE)
    title = "mAP Values per Run with Standard Deviation"
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.yticks(fontsize = TEXT_FONTSIZE)
    plt.xticks(x_indices, run_names, ha="right", fontsize=TEXT_FONTSIZE)
    plt.legend(fontsize=TEXT_FONTSIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_path = output_folder.joinpath(title+'.png')
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    plt.close()
    print(f'Saved {save_path}')

def plot(root_folder:str|Path, output_folder:str|Path):
    data = process_folder(root_folder)
    plot_metrics(data, output_folder)
    plot_map(root_folder, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CSV files representing  K-fold Cross Validation training and evaluation data.")
    parser.add_argument('root', type=Path, help='Root folder for all CSV files.')
    parser.add_argument('--output', type=Path, default=None, required=False, help='Output folder for the plot files. Default [root]')
    args = parser.parse_args()
    out = args.output if args.output is not None else args.root
    plot(args.root, out)