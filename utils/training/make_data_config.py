"""Make the training, validation, and test sets from a YOLO formatted dataset.

Raises:
    FileNotFoundError: Could not find the 'images' folder of the dataset.
    TypeError: Wrong a wrong split has been provided.
"""
import argparse
import os

from pathlib import Path

SEED = 42
TRAIN_SPLIT_FILENAME = "train.txt"
VALIDATION_SPLIT_FILENAME = "validation.txt"
TEST_SPLIT_FILENAME = "test.txt"

def __partition(elements:list, proportions:list[float]) -> list[list]:
    total = sum(proportions)
    proportions = [p/total for p in proportions]
    
    result = []
    start = 0
    n = len(elements)

    for i, proportion in enumerate(proportions):
        # for the last slice, take all remaining elements to avoid rounding gaps
        if i == len(proportions) - 1:
            result.append(elements[start:])
        else:
            end = start + round(proportion * n)
            result.append(elements[start:end])
            start = end

    return result

def make_data_config(images_dir:Path|str, output_dir:Path|str, shuffle:bool, split:list[float]) -> list[Path]:
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    if len(split) not in [2, 3]:
        raise TypeError(f'--split must have 2 or 3 arguments, provided: {args.split}')
    
    total_proportion = sum(split)
    split = [part/total_proportion for part in split]

    images = list(images_dir.glob("*.*"))

    if shuffle:
        import random
        random.seed(SEED)
        random.shuffle(images)
    
    partitioned_images = __partition(images, split)

    train_samples = partitioned_images[0]
    validation_samples = partitioned_images[1]        
    

    with open(output_dir.joinpath(TRAIN_SPLIT_FILENAME), 'w') as f:
        f.writelines([str(s)+'\n' for s in train_samples])
    with open(output_dir.joinpath(VALIDATION_SPLIT_FILENAME), 'w') as f:
        f.writelines([str(s)+'\n' for s in validation_samples])

    output_files = [
        output_dir.joinpath(TRAIN_SPLIT_FILENAME).absolute(),
        output_dir.joinpath(VALIDATION_SPLIT_FILENAME).absolute()
    ]

    if len(partitioned_images) == 3:
        test_images = partitioned_images[2]
        with open(output_dir.joinpath(TEST_SPLIT_FILENAME), 'w') as f:
            f.writelines([str(s)+'\n' for s in test_images])
        output_files.append(output_dir.joinpath(TEST_SPLIT_FILENAME).absolute())
    
    return output_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove YOLO annotations by class ID.")
    parser.add_argument("images", type=Path, help="Directory containing the image samples.")
    parser.add_argument("--output", type=Path, required=False, default=None, help="Output directory of data configuration files. (default: dataset root)")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False, help="Randomize samples before splitting.")
    parser.add_argument('--split', nargs='+', type=float, default=[0.8, 0.2], help='Sample split proportions [train, validation] or [train, validation, test].')
    args = parser.parse_args()

    # check arguments
    images:Path = args.images

    if not images.exists():
        raise FileNotFoundError(f'Images directory not found: {images.absolute()}')
    
    output_dir = args.output if args.output is not None else images.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    split:list = args.split

    make_data_config(images, output_dir, args.shuffle, split)
