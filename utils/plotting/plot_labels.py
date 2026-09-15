import os, math
import matplotlib.pyplot as plt
from pathlib import Path

from ..training import my_logging
from ..labels import label_analysis

LOGS_FOLDER = "logs/plotting_logs"
PLOT_FOLDER = os.path.join("plots")

# since there are a lot of models, we need something to differentiates the various lines
LINE_STYLES = ['solid', "dotted", (0, (3, 1, 1, 1, 1, 1)), "dashed", 'dashdot'] * 5

# colour palette (colorblind safe, Paul Tol's Bright)
PALETTE = [
    '#4477AA',
    '#EE6677',
    '#228833',
    '#CCBB44',
    '#66CCEE',
    '#AA3377',
    '#BBBBBB',
] * 2
FIGURE_SIZE = (16, 9)
TITLE_FONTSIZE = 24
TEXT_FONTSIZE = 20


def get_logger():
    return my_logging.get_logger("PlotsLogger", my_logging.logging.INFO, LOGS_FOLDER)

def create_folder(folder:str|Path|None):
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)


def plot_labels_distribution(datasets:list[str]|list[Path], show:bool=False, out_folder:str|Path|None = None):
    logger = get_logger()
    logger.info(f"Plotting dataset stats for: {datasets}")
    logger.debug(f"Output folder: {out_folder}")
    create_folder(out_folder)

    plot_title = "Label Distribution"
    logger.debug(plot_title)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for i, dataset_folder in enumerate(datasets):
        dataset = Path(dataset_folder)
        dataset_name = dataset.stem
        logger.debug(f"Add label distribution for {dataset_name}")

        labels_folder = label_analysis.get_labels_folder(dataset_folder)
        if labels_folder is None:
            logger.error(f"Folder {labels_folder} not found")
            continue

        stat = label_analysis.analyze_instances_distribution(labels_folder)

        if stat is None:
            logger.error(f"Could not obtain label distribution for dataset {dataset_name}")
            continue

        ax.plot(stat, label = dataset_name, linestyle=LINE_STYLES[i], color=PALETTE[i])
    plt.grid(visible=True)
    ax.set_ybound(upper=150)
    ax.set_xbound(upper=200)
    ax.set_xlabel("Instances present in the image")
    ax.set_ylabel("Number of images")
    ax.set_title(plot_title)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    if out_folder is not None:
        path = f"{out_folder}/{plot_title}.png"
        logger.debug(f"Saving image to path: {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        logger.debug("Showing plot")
        plt.show()
    plt.close()

def plot_instances(datasets:list[str]|list[Path], show:bool=False, out_folder:str|Path|None = None):
    logger = get_logger()
    logger.info("Plotting the number of instances per dataset")
    logger.debug(str(datasets))
    logger.debug(f"Output folder: {out_folder}")
    create_folder(out_folder)

    plot_title = "Number of item instances per dataset"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    for i, dataset_path in enumerate(datasets):
        dataset = Path(dataset_path)
        dataset_name = dataset.stem
        logger.debug(f"Add instance count for {dataset_name}")

        labels_folder = label_analysis.get_labels_folder(dataset)
        if labels_folder is None:
            logger.error(f"Folder {labels_folder} not found")
            continue

        stat = label_analysis.number_of_instances_in_folder(labels_folder)

        if stat is None:
            continue

        container = ax.bar(dataset_name, stat, linestyle=LINE_STYLES[i], color=PALETTE[i])
        ax.bar_label(container,fmt='{:,.0f}')

    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Number of instances", fontsize=TEXT_FONTSIZE)
    # ax.set_xlabel(fontsize=TEXT_FONTSIZE)
    # plt.yticks(fontsize = TEXT_FONTSIZE)
    # plt.xticks(fontsize = TEXT_FONTSIZE)
    plt.grid(visible=True,axis='y')

    if out_folder is not None:
        path = f"{out_folder}/{plot_title}.png"
        logger.debug(f"Saving image to path: {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        logger.debug("Showing plot")
        plt.show()

    plt.close()

def plot_sizes(datasets:list[str]|list[Path], show:bool=False, out_folder:str|Path|None = None):
    plot_mean_widths(datasets,show,out_folder)
    plot_mean_heights(datasets,show,out_folder)
    plot_mean_areas(datasets,show,out_folder)

def plot_mean_widths(datasets:list[str]|list[Path], show:bool=False, out_folder:str|Path|None = None):
    logger = get_logger()
    logger.info("Drawing plots for the width of labels")
    logger.debug(str(datasets))
    logger.debug(f"Output folder: {out_folder}")
    create_folder(out_folder)

    plot_title = "Labels width distribution as percentage of image"

    datasets_names = [Path(d).stem for d in datasets]

    widths = list()
    for dataset in datasets:
        labels_folder = label_analysis.get_labels_folder(dataset)
        if labels_folder is None:
            logger.error(f"Folder {labels_folder} not found")
            continue

        labels_sizes = label_analysis.get_labels_sizes(labels_folder)
        if labels_sizes is None:
            continue
        widths.append([w for (w,h) in labels_sizes])
    

    means = list()
    standard_deviations = list()
    for widths_data in widths:
        mean = sum(widths_data)/len(widths_data)
        deltas = [math.pow((width - mean), 2) for width in widths_data]
        deviation = math.sqrt(sum(deltas) / len(deltas))
        means.append(mean)
        standard_deviations.append(deviation)

    violins_labels = [dataset_name + "\n" + "μ:{:1.4f}\nσ:{:1.4f}".format(mean, variance) for dataset_name, mean, variance in zip(datasets_names, means, standard_deviations)]


    fig_h, ax_w = plt.subplots(figsize=FIGURE_SIZE)
    ax_w.set_yticks([y + 1 for y in range(len(datasets_names))], labels=violins_labels)
    violins = ax_w.violinplot(widths, showmeans=True, orientation="horizontal")
    
    colors = PALETTE[:len(datasets_names)]
    colors.reverse()
    for violin, color in zip(violins["bodies"], colors): # type: ignore
        violin.set_facecolor(color)
        violin.set_alpha(0.9)

    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violins[partname]
        vp.set_edgecolor('Black')
        vp.set_linewidth(1)

    plt.grid(visible=True)
    ax_w.set_xlabel("Label width distribution per dataset")
    # ax_w.set_title(plot_title)

    if out_folder is not None:
        path = f"{out_folder}/{plot_title}.png"
        logger.debug(f"Saving image to path: {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        logger.debug("Showing plot")
        plt.show()
    
    plt.close()

def plot_mean_areas(datasets:list[str]|list[Path], show:bool=False, out_folder:str|Path|None = None):
    logger = get_logger()
    logger.info("Drawing plots for the areas of labels")
    local_datasets = datasets.copy()
    logger.debug(str(local_datasets))
    logger.debug(f"Output folder: {out_folder}")
    create_folder(out_folder)

    plot_title = "Labels area distribution as percentage of image per dataset."
    local_datasets.reverse()
    datasets_names = [Path(d).stem for d in local_datasets]

    areas = list()
    for dataset in local_datasets:
        labels_folder = label_analysis.get_labels_folder(dataset)
        if labels_folder is None:
            logger.error(f"Folder {labels_folder} not found")
            continue

        labels_sizes = label_analysis.get_labels_sizes(labels_folder)
        if labels_sizes is None:
            continue
        areas.append([w*h for (w,h) in labels_sizes])

    means = list()
    standard_deviations = list()
    for areas_data in areas:
        mean = sum(areas_data)/len(areas_data)
        deltas = [math.pow((width - mean), 2) for width in areas_data]
        deviation = math.sqrt(sum(deltas) / len(deltas))
        means.append(mean)
        standard_deviations.append(deviation)

    fig_h, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_yticks([y + 1 for y in range(len(datasets_names))], labels=datasets_names, fontsize=TEXT_FONTSIZE)
    violins = ax.violinplot(areas, showmeans=True, orientation="horizontal")
    
    colors = PALETTE[:len(datasets_names)]
    colors.reverse()
    for violin, color in zip(violins["bodies"], colors): # type: ignore
        violin.set_facecolor(color)
        violin.set_alpha(0.9)

    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violins[partname]
        vp.set_edgecolor('Black')
        vp.set_linewidth(1)

    plt.grid(visible=True, axis='x')
    ax.set_xbound(-0.001, 0.03)
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(r"Labels' area as % of the whole image.", fontsize=TEXT_FONTSIZE)

    import matplotlib.ticker as mtick
    tick_format = mtick.PercentFormatter(symbol="%")
    ax.xaxis.set_major_formatter(tick_format)

    plt.xticks(fontsize = TEXT_FONTSIZE)
    if out_folder is not None:
        path = f"{out_folder}/{plot_title}.png"
        logger.debug(f"Saving image to path: {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        logger.debug("Showing plot")
        plt.show()
    
    plt.close()

def plot_mean_heights(datasets:list[str]|list[Path], show:bool=False, out_folder:str|Path|None = None):
    logger = get_logger()
    logger.info("Drawing plots for height of labels")
    logger.debug(str(datasets))
    logger.debug(f"Output folder: {out_folder}")
    create_folder(out_folder)

    plot_title = "Labels height distribution as percentage of image"
    datasets_names = [Path(d).stem for d in datasets]
    heights : list[list[float]] = list()
    
    for dataset in datasets:
        labels_folder = label_analysis.get_labels_folder(dataset)
        if labels_folder is None:
            logger.error(f"Folder {labels_folder} not found")
            continue

        labels_sizes = label_analysis.get_labels_sizes(labels_folder)

        if labels_sizes is None:
            continue
        heights.append([h for (w,h) in labels_sizes])
    
    means = list()
    standard_deviations = list()
    for heights_data in heights:
        mean = sum(heights_data)/len(heights_data)
        deltas = [math.pow((height - mean), 2) for height in heights_data]
        deviation = math.sqrt(sum(deltas) / len(deltas))
        means.append(mean)
        standard_deviations.append(deviation)
    
    violins_labels = [dataset_name + "\n" + "μ:{:1.4f}\nσ:{:1.4f}".format(mean, variance) for dataset_name, mean, variance in zip(datasets_names, means, standard_deviations)]
    
    fig_h, ax_h = plt.subplots(figsize=FIGURE_SIZE)
    ax_h.set_xticks([y + 1 for y in range(len(datasets_names))], labels=violins_labels)
    violins = ax_h.violinplot(heights, showmeans=True, orientation="vertical")
    
    for violin, color in zip(violins["bodies"], PALETTE): # type: ignore
        violin.set_facecolor(color)
        violin.set_alpha(0.9)
    
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violins[partname]
        vp.set_edgecolor('Black')
        vp.set_linewidth(1)

        
    plt.grid(visible=True)
    ax_h.set_ylabel("Label height distribution per dataset")
    ax_h.set_title(plot_title)

    if out_folder is not None:
        path = f"{out_folder}/{plot_title}.png"
        logger.debug(f"Saving image to path: {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        logger.debug("Showing plot")
        plt.show()
    
    plt.close()

def plot_instances_per_image(datasets:list[str]|list[Path], show:bool=False, out_folder:str|Path|None = None):
    logger = get_logger()
    logger.info("Drawing plot for the mean number of instances per image from the datasets")
    logger.debug(str(datasets))
    logger.debug(f"Output folder: {out_folder}")
    create_folder(out_folder)

    plot_title = 'Distribution of item instances per image per dataset'
    datasets_names = [Path(d).stem for d in datasets]

    instances_per_image = list()
    for dataset in datasets:
        labels_folder = label_analysis.get_labels_folder(dataset)
        if labels_folder is None:
            logger.error(f"Folder {labels_folder} not found")
            continue

        instances_list = label_analysis.instances_per_image_in_folder(labels_folder)
        instances_per_image.append(instances_list)

    means = list()
    standard_deviations = list()
    for instances_data in instances_per_image:
        mean = sum(instances_data)/len(instances_data)
        deltas = [math.pow((instances - mean), 2) for instances in instances_data]
        deviation = math.sqrt(sum(deltas) / len(deltas))
        means.append(mean)
        standard_deviations.append(deviation)

    for dataset_name, m, mu, in zip(datasets_names, means, standard_deviations):
        logger.info(f'{dataset_name}: Mean {m}, Standard Deviation {mu}')
        
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_xticks([y + 1 for y in range(len(datasets_names))], labels=datasets_names)
    violins = ax.violinplot(instances_per_image, showmeans=True, orientation="vertical")

    for violin, color in zip(violins["bodies"], PALETTE): # type: ignore
        violin.set_facecolor(color)
        violin.set_alpha(0.9)
    
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violins[partname]
        vp.set_edgecolor('Black')
        vp.set_linewidth(1)
    
    plt.grid(visible=True, axis='y')
    plt.yticks(fontsize=TEXT_FONTSIZE)
    ax.set_ylabel("Number of item instances", fontsize=TEXT_FONTSIZE)
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)

    if out_folder is not None:
        path = f"{out_folder}/{plot_title}.png"
        logger.debug(f"Saving image to path: {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        logger.debug("Showing plot")
        plt.show()
    
    plt.close()

def plot_number_of_images_per_dataset(datasets:list[str]|list[Path], show:bool=False, out_folder:str|Path|None = None):
    logger = get_logger()
    logger.info("Drawing plot for the total number of images in the datasets")
    logger.debug(str(datasets))
    logger.debug(f"Output folder: {out_folder}")
    create_folder(out_folder)

    plot_title = "Number of images per dataset"
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    dataset_names = [Path(d).stem for d in datasets]
    first_container_labels = None
    for i, (dataset_name, dataset) in enumerate(zip(dataset_names, datasets)):
        labels_folder = label_analysis.get_labels_folder(dataset)
        if labels_folder is None:
            logger.error(f"Folder {labels_folder} not found")
            continue

        data = label_analysis.get_number_of_images(labels_folder,47)
        if data is None:
            continue
        container_labels = ax.bar(dataset_name, data[0], linestyle=LINE_STYLES[i], color=PALETTE[i])
        first_container_labels = container_labels if first_container_labels is None else first_container_labels
        ax.bar_label(container_labels, fmt = str(sum(data)))

    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Number of images", fontsize=TEXT_FONTSIZE)
    # ax.set_xlabel(fontsize=TEXT_FONTSIZE)
    # plt.yticks(fontsize = TEXT_FONTSIZE)
    # plt.xticks(fontsize = TEXT_FONTSIZE)

    # ax.set_ybound(upper=2500)
    plt.grid(visible=True,axis='y')

    if out_folder is not None:
        path = f"{out_folder}/{plot_title}.png"
        logger.debug(f"Saving image to path: {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        logger.debug("Showing plot")
        plt.show()
    
    plt.close()
