"""
draw_box.py
Draws the annotations of a YOLO like dataset.

Usage
-----
python draw_box.py --dataset path/to/root_dataset_directory --output path/to/annotated_images_directory
python draw_box.py --images path/to/images --labels path/to/labels --classes path/to/classes.txt --output path/to/annotated_images_directory

"""
import cv2, random, argparse
from pathlib import Path
from tqdm import tqdm

def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    thickness = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, thickness / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_boxes(image_path:Path, classes:list[str], colors, labels_dir:Path, output_dir:Path) -> int:
    
    annotation_path = labels_dir.joinpath(image_path.stem+'.txt')
    image = cv2.imread(str(image_path))

    if image is None:
        print(f'Error: could not open image {image_path}.')
        return 0
    if not annotation_path.exists():
        print(f'Annotation file "{annotation_path} does not exists.')
        return 0
    
    try:
        height, width, channels = image.shape
    except:
        print(f'no shape info for image {image_path}.')
        return 0

    box_number = 0
    with open(annotation_path) as annotation_file:
        for line in annotation_file:
            staff = line.split()
            class_idx = int(staff[0])

            x_center, y_center, w, h = float(
                staff[1])*width, float(staff[2])*height, float(staff[3])*width, float(staff[4])*height
            x1 = round(x_center-w/2)
            y1 = round(y_center-h/2)
            x2 = round(x_center+w/2)
            y2 = round(y_center+h/2)

            plot_one_box([x1, y1, x2, y2], image, color=colors[class_idx], label=classes[class_idx], line_thickness=None)

            box_number += 1
    
    save_file_path = output_dir.joinpath(image_path.name)
    cv2.imwrite(str(save_file_path), image)
    return box_number

def classes_from_file(file:Path):
    lines = list()
    with open(file) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines

def draw_folder(images_dir:Path|str, labels_dir:Path|str, classes_file:Path|str, output_dir:Path|str):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    classes_file = Path(classes_file)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    classes = classes_from_file(classes_file)

    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    images = images_dir.glob(pattern='*.*')
    number_of_images = len(list(images_dir.glob(pattern='*.*')))
    box_total = 0
    image_total = 0
    for image in tqdm(images, 'Drawing labels', number_of_images):
        box_num = draw_boxes(image, classes, colors, labels_dir, output_dir)
        box_total += box_num
        image_total += 1
    print('Number of boxes:', box_total, 'Number of images:', image_total)

def draw_dataset(dataset:Path|str, output:Path|str):
    dataset = Path(dataset)
    output = Path(output)
    images_folder = dataset / 'images'
    labels_folder = dataset / 'labels'
    classes = dataset / 'classes.txt'
    draw_folder(images_folder, labels_folder, classes, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw bounding boxes of a YOLO-like dataset.")
    parser.add_argument("--dataset", type=Path, required=False, default=None, help="Root directory of a yolo dataset.")
    parser.add_argument("--images", type=Path, required=False, help="Directory of the images")
    parser.add_argument("--labels", type=Path, required=False, help="Directory of the labels")
    parser.add_argument("--classes", type=Path, required=False, help="Classes file")
    parser.add_argument("--output", type=Path, help="Output directory")
    args = parser.parse_args()

    images_folder = args.images
    labels_folder = args.labels
    classes = args.classes
    if args.dataset is not None:
        draw_dataset(args.dataset, args.output)
    else:
        draw_folder(images_folder, labels_folder, classes, args.output)

