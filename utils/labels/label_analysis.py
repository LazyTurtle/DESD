import math, argparse
from pathlib import Path

from ..training import my_logging

LOGS_FOLDER = "logs/labels_analysis"


class IterativeCalculator:

    def __init__(self, name:str):
        self.setup(name)
    
    def setup(self, name:str):
        self.logger = my_logging.get_logger(name, my_logging.logging.INFO, LOGS_FOLDER)
        self.logger.info(f"Setup iterative calculator for value: {name}")
        self.name = name
        self.initialized = False
        self.count = 0
        self.M2 = 0.0
        self.mean = None
    
    def add_number(self, number):
        self.logger.debug(f"Adding number {number}")
        self.count += 1

        if self.initialized:
            old_mean = self.mean
            delta_old = number - old_mean
            self.mean += delta_old / self.count
            delta_new = number - self.mean
            self.M2 += delta_old * delta_new
        else:
            self.initialized = True
            self.mean = number

    @property
    def variance(self):
        return self.M2 / self.count

    @property
    def std(self):
        return math.sqrt(self.variance)    
    
    def reset(self, name:str):
        self.setup(name)


def get_logger():
    return my_logging.get_logger("Analysis", my_logging.logging.INFO, LOGS_FOLDER)

def get_labels_folder(path:str|Path) -> Path | None:
    path = Path(path)
    
    last_folder = path.parent if path.is_file() else path

    match last_folder.stem:
        case 'labels':
            return path if path.exists() else None
        case 'image':
            new_path = path.parent / 'labels'
            return new_path if new_path.exists() else None
        case _:
            new_path = path / 'labels'
            return new_path if new_path.exists() else None

def get_images_folder(path:str|Path) -> Path | None:
    path = Path(path)
    
    last_folder = path.parent if path.is_file() else path

    match last_folder.stem:
        case 'images':
            return path if path.exists() else None
        case 'labels':
            new_path = path.parent / 'images'
            return new_path if new_path.exists() else None
        case _:
            new_path = path / 'images'
            return new_path if new_path.exists() else None


def get_number_of_images(folder:str|Path, class_id:int|None = None)->tuple[int, int] | None:
    folder = Path(folder)
    logger = get_logger()
    logger.info(f"Calculating the number of instances")
    labels_folder = get_labels_folder(folder)
    if labels_folder is None:
        logger.error(f"The folder provided is not recognized: {labels_folder}")
        return None
    
    logger.info(f"Folder {labels_folder}")
    logger.info(f"Class selected: {class_id}")
    annotation_files = labels_folder.glob('*.txt')
    non_background_images = 0
    background_images = 0
    for annotation_file in annotation_files:
        with open(annotation_file) as file:
            lines = file.readlines()
            lines = [l for l in lines if l != '\n']
            if class_id is not None:
                lines = [l for l in lines if int(l.split()[0]) == class_id]
            if len(lines) > 0:
                non_background_images += 1
            else:
                background_images += 1
    logger.info(f'Non background images: {non_background_images}')
    logger.info(f'background images: {background_images}')
    return (non_background_images, background_images)
            
def avg_items_per_image(folder:str, class_id:int|None = None)->float | None:
    logger = get_logger()
    logger.info('Collecting average number of items per image')
    labels_folder = get_labels_folder(folder)
    if labels_folder is None:
        logger.error(f"The folder provided is not recognized: {labels_folder}")
        return None

    logger.info(f"Folder: {labels_folder}")
    logger.info(f"Selected class: {class_id}")
    annotation_files = labels_folder.glob('*.txt')
    items_per_image_calculator = IterativeCalculator("Items per image")
    for file in annotation_files:
        items = count_items(file, class_id)
        if items is not None:
            items_per_image_calculator.add_number(items)

    mean = items_per_image_calculator.mean
    variance = items_per_image_calculator.variance

    logger.info(f"Average items per image: {mean}")
    logger.info(f"Variance: {variance}")
    return mean


def count_items(file:str|Path, class_id:int|None = None)->int|None:
    file = Path(file)
    logger = get_logger()
    logger.debug(f"Reading file: {file}")
    logger.debug(f"Selecting class: {class_id}")

    lines = 0
    try:
        with open(file, 'r') as fl:
            items = fl.readlines()
            items = [item for item in items if item != '\n']
            if class_id is not None:
                class_lines = 0
                for item in items:
                    line_elements = item.split()
                    if int(line_elements[0]) == class_id:
                        class_lines += 1
                lines = class_lines
            else:
                lines = len(items)
    except:
        lines = None
    
    if lines is None:
        logger.error("Could not read the lines")
    else:
        logger.debug(f"Number of items: {lines}")
    return lines


def stats_size_of_items(folder:str, class_id:int|None = None) -> tuple[tuple[float, float],tuple[float, float],tuple[float, float]] | None:
    logger = get_logger()
    logger.info(f"Collecting the average sizes of items")

    labels_folder = get_labels_folder(folder)
    if labels_folder is None:
        logger.error(f"The folder provided is not recognized: {labels_folder}")
        return None

    logger.info(f"Folder: {labels_folder}")
    logger.info(f"Selected class: {labels_folder}")

    annotation_files = labels_folder.glob('*.txt')

    width_calculator = IterativeCalculator("width")
    height_calculator = IterativeCalculator("height")
    area_calculator = IterativeCalculator("area")

    try:
        for file in annotation_files:
            with open(file) as f:
                logger.debug(f"Reading file: {file}")
                lines = f.readlines()
                for line in lines:
                    if line == '\n':
                        continue
                    line_elements = line.split()
                    label_class = int(line_elements[0])
                    if class_id is not None and label_class != class_id:
                        continue
                    width = float(line_elements[3])
                    height = float(line_elements[4])

                    width_calculator.add_number(width)
                    height_calculator.add_number(height)
                    area_calculator.add_number(width * height)
    except Exception as e:
        logger.error(str(e))
        return None
    
    mean_width, variance_width = width_calculator.mean, width_calculator.variance
    mean_height, variance_height = height_calculator.mean, height_calculator.variance
    mean_area, variance_area = area_calculator.mean, area_calculator.variance

    logger.info(f"Width - mean: {mean_width}, variance: {variance_width}")
    logger.info(f"Height - mean: {mean_height}, variance: {variance_height}")
    logger.info(f"Area - mean: {mean_area}, variance: {variance_area}")

    result = None
    if mean_width and mean_height and mean_area:
        result = ((mean_width, variance_width), (mean_height, variance_height), (mean_area, variance_area))
        
    return result
    


def number_of_instances_in_folder(folder:str|Path, class_id:int|None = None) -> int | None:
    folder = Path(folder)
    logger = get_logger()
    logger.info(f"Calculating the number of instances")

    labels_folder = get_labels_folder(folder)
    if labels_folder is None:
        logger.error(f"The folder provided is not recognized: {labels_folder}")
        return None

    logger.info(f"Folder {labels_folder}")
    logger.info(f"Class selected: {class_id}")
    annotation_files = labels_folder.glob('*.txt')
    instances = 0
    for file in annotation_files:
        count_of_image = count_items(file, class_id)

        if count_of_image is None:
            logger.error(f"Count for image {file} is None")
            continue

        instances += count_of_image
    logger.info(f"Number of items: {instances}")

    return instances

def instances_per_image_in_folder(folder:str|Path, class_id:int|None = None) -> list[int] | None:
    folder = Path(folder)
    logger = get_logger()
    logger.info(f"Collecting the number of instances per image")
    labels_folder = get_labels_folder(folder)
    if labels_folder is None:
        logger.error(f"The folder provided is not recognized: {labels_folder}")
        return None

    logger.info(f"Folder {labels_folder}")
    logger.info(f"Class selected: {class_id}")
    annotation_files = labels_folder.glob('*.txt')

    instances_per_image = list()
    for file in annotation_files:
        instances = count_items(file, class_id)
        if instances is None:
            logger.error(f"Count for image {file} is None")
            continue
        instances_per_image.append(instances)
    logger.info(f'Total number of images: {len(instances_per_image)}')
    logger.info(f'Total number of instances: {sum(instances_per_image)}')
    return instances_per_image
    

def min_max_number_of_items(folder:str|Path, class_id:int|None = None) -> tuple[int | None, int | None] | None:
    folder = Path(folder)
    logger = get_logger()
    labels_folder = get_labels_folder(folder)
    if labels_folder is None:
        logger.error(f"The folder provided is not recognized: {labels_folder}")
        return None

    logger.info(f"Calculate the maximum number of items in an image from folder {labels_folder}")
    logger.info(f"Class selected: {class_id}")
    annotation_files = labels_folder.glob('*.txt')
    maximum = float("-inf")
    minimum = float("inf")

    for file in annotation_files:
        instances = count_items(file, class_id)
        maximum = instances if (instances is not None) and (instances > maximum) else maximum
        minimum = instances if (instances is not None) and (instances < minimum) else minimum

    maximum = int(maximum) if maximum > float("-inf") else None
    minimum = int(minimum) if minimum < float("inf") else None

    logger.info(f"Maximum number of instances per image: {maximum}")
    logger.info(f"Minimum number of instances per image: {minimum}")
    return (minimum, maximum)


def analyze_instances_distribution(folder:str|Path, class_id:int|None = None) -> list[int] | None:
    folder = Path(folder)
    logger = get_logger()
    logger.info("Calculating label distribution")
    labels_folder = get_labels_folder(folder)
    if labels_folder is None:
        logger.error(f"The folder provided is not recognized: {labels_folder}")
        return None
    logger.info(f"Folder: {labels_folder}")
    logger.info(f'Class selected: {class_id}')

    min_max = min_max_number_of_items(labels_folder, class_id)
    if min_max is None:
        return None
    
    minimum, maximum = min_max
    if maximum is None or minimum is None:
        logger.error("Could not find the minimum or maximum number of items")
        return
    
    label_distribution = [0] * (maximum + 1)
    annotation_files = labels_folder.glob('*.txt')
    for file in annotation_files:
        instances = count_items(file, class_id)
        if instances is not None:
            label_distribution[instances] += 1
    
    logger.info(f"Distribution obtained: {label_distribution}")
    return label_distribution

def get_labels_sizes(folder:str|Path, class_id:int|None = None) -> list[tuple[float, float]] | None:
    folder = Path(folder)
    logger = get_logger()
    logger.info("Collecting all annotations size data")
    labels_folder = get_labels_folder(folder)
    if labels_folder is None:
        logger.error(f"The folder provided is not recognized: {labels_folder}")
        return None

    logger.info(f"Folder: {labels_folder}")
    logger.info(f'Class selected: {class_id}')

    annotation_files = labels_folder.glob('*.txt')
    annotations_sizes = list()
    for annotation_file in annotation_files:
        logger.debug(f"Reading file: {annotation_file}")
        labels_added = 0
        try:
            with open(annotation_file) as file:
                lines = file.readlines()
                for line in lines:
                    elements = line.split()
                    label_class = int(elements[0])
                    if class_id is not None and label_class != class_id:
                        continue
                    width = float(elements[3])
                    height = float(elements[4])
                    annotations_sizes.append((width, height))
                    labels_added += 1
        except Exception as e:
            logger.error(str(e))
        logger.debug(f"Added {labels_added} labels from file {annotation_file}")
    logger.info(f"Labels size data collected: {len(annotations_sizes)}")
    return annotations_sizes


def analyze_folder(folder:str, class_id:int|None = None):
    avg_items_per_image(folder, class_id)
    stats_size_of_items(folder, class_id)
    analyze_instances_distribution(folder, class_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the labels of the provided YOLO dataset.")
    parser.add_argument('root', type=Path, help='Root folder of the dataset.')
    parser.add_argument('--id', type=int, required=False, default=None, help='When provided, limits the analysis on a single class. Default None.')
    args = parser.parse_args()
    analyze_folder(args.root, args.id)