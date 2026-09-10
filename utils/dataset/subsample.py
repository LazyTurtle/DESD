
import shutil
from pathlib import Path
from tqdm import tqdm
from random import shuffle

def reduce_dataset(dataset: str|Path, output_dataset: str|Path, n:int, overwrite:bool=False) -> list[Path]:
    """
    Copy N alphabetically ordered, equally spaced samples from a source dataset to a new dataset.

    Returns the list of copied filenames.
    """
    def copy_n_equally_spaced_files(source:Path, destination:Path, n:int):
        destination.mkdir(parents=True, exist_ok=True)
        files = sorted(list(source.glob('*.*')))
        if n >= len(files):
            selected_files = files
        else:
            step = (len(files) - 1) / (n - 1)
            indices = [round(i * step) for i in range(n)]
            selected_files = [files[i] for i in indices]

        for file in tqdm(selected_files, ):
            shutil.copy2(file, destination / file.name)

        return selected_files

    if n <= 0:
        return []
    dataset = Path(dataset)
    output_dataset = Path(output_dataset)

    if output_dataset.exists() and not overwrite:
        print(f'Dataset {output_dataset} already exists.')
        return []
    if output_dataset.exists():
        shutil.rmtree(output_dataset)
    
    in_images = dataset / 'images'
    in_labels = dataset / 'labels'
    out_images = output_dataset / 'images'
    out_labels = output_dataset / 'labels'
    copied_images = copy_n_equally_spaced_files(in_images, out_images, n)
    copied_labels = copy_n_equally_spaced_files(in_labels, out_labels, n)
    print(f'Copied {len(copied_images)} images from {in_images} to {out_images}')
    print(f'Copied {len(copied_labels)} labels from {in_labels} to {out_labels}')

    return copied_images

def subsample_dataset(dataset: str|Path, output_dataset: str|Path, n:int, overwrite:bool=False) -> list[Path]:

    def copy_every_n_file(source:Path, destination:Path, n:int):
        destination.mkdir(parents=True, exist_ok=True)

        files = sorted(list(source.glob('*.*')))
        copied = list()
        for i, file in tqdm(enumerate(files)):
            if i % n == 0:

                dest = destination / file.name
                shutil.copy2(file, dest)
                copied.append(dest)
        return copied

    dataset = Path(dataset)
    output_dataset = Path(output_dataset)
    if output_dataset.exists() and not overwrite:
        print(f'Dataset {output_dataset} already exists.')
        return []
    if output_dataset.exists():
        shutil.rmtree(output_dataset)

    in_images = dataset / 'images'
    in_labels = dataset / 'labels'
    out_images = output_dataset / 'images'
    out_labels = output_dataset / 'labels'
    copied_images = copy_every_n_file(in_images, out_images, n)
    copied_labels = copy_every_n_file(in_labels, out_labels, n)
    print(f'Copied {len(copied_images)} images from {in_images} to {out_images}')
    print(f'Copied {len(copied_labels)} labels from {in_labels} to {out_labels}')
    return copied_images

def random_subsample_dataset(dataset: str|Path, output_dataset: str|Path, n:int, overwrite:bool=False) -> list[Path]:

    dataset = Path(dataset)
    output_dataset = Path(output_dataset)
    if output_dataset.exists() and not overwrite:
        print(f'Dataset {output_dataset} already exists.')
        return []
    if output_dataset.exists():
        shutil.rmtree(output_dataset)

    in_images = dataset / 'images'
    in_labels = dataset / 'labels'
    out_images = output_dataset / 'images'
    out_labels = output_dataset / 'labels'
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = list(in_images.glob('*.*'))
    shuffle(images)
    images_to_copy = images[:n]
    out = list()
    for image in tqdm(images_to_copy):

        dest_img = out_images / image.name
        shutil.copy2(image, dest_img)
        out.append(dest_img)

        label = in_labels / (image.stem + ".txt")
        shutil.copy2(label, out_labels / label.name)
    return out
