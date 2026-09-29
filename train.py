import os, sys, yaml, argparse, torch, gc, random

from ultralytics import YOLO
from pathlib import Path

from utils.training import my_logging

HOME = os.getcwd().strip()
LOG_FOLDER = r'logs\train'

def logger():
    return my_logging.get_logger('main_logger', out_folder=LOG_FOLDER)

def clean_memory():
    LOGGER = logger()
    LOGGER.debug('Cleaning memory...')
    gc.collect()
    torch.cuda.empty_cache()
    LOGGER.debug('Memory cleaned')

def load_config(config_path:Path|str)->dict[str,str]:
    LOGGER = logger()
    yaml_configuration_file = Path(config_path)
    if not yaml_configuration_file.exists():
        err = f'Hyperparameter {yaml_configuration_file} file does not exist.'
        LOGGER.error(err)
        sys.exit(err)

    LOGGER.info('Training from yaml configuration file.')
    LOGGER.debug(f'Yaml: {yaml_configuration_file}')

    config = None
    with open(yaml_configuration_file, 'r') as config_file:
        LOGGER.debug('Reading yaml config file')
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    if config is None:
        LOGGER.error('Could not read configuration file')
        raise RuntimeError('Could not read configuration file')
    
    LOGGER.debug('Cleaning config yaml')
    none_parameters = [key for key, item in config.items() if item == 'None']
    LOGGER.debug(f'Removing "None" parameters: {none_parameters}')
    config = {key:value for key, value in config.items() if value != 'None'}

    if 'project' in config.keys():
        new_project_folder = os.path.join(HOME, config['project'])
        LOGGER.debug(f'Updating project folder from "{config["project"]}" to "{new_project_folder}"')
        config['project'] = new_project_folder
    
    LOGGER.debug('Final config')
    LOGGER.debug((str(config)))
    return config

def load_samples(dataset:Path) -> list[Path]:
    images = dataset / 'images'
    samples = list(images.glob('*.*'))
    return samples

def samples_to_file(samples:list[Path], file_path:Path):
    with open(file_path, 'w') as f:
        lines = [str(s.absolute())+'\n' for s in samples]
        f.writelines(lines)


def get_kfold_splits(samples, k=5, shuffle=True, seed=None):
    if k <= 1:
        raise ValueError(f"k={k} must be greater than 1.")
    if k > len(samples):
        raise ValueError(f"k={k} cannot be greater than the total number of samples={len(samples)}.")
    
    paths = list(samples)
    
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(paths)
        
    n_samples = len(paths)
    
    base_size = n_samples // k
    remainder = n_samples % k
    
    fold_sizes = [base_size + (1 if i < remainder else 0) for i in range(k)]
    
    start_idx = 0
    for fold_size in fold_sizes:
        end_idx = start_idx + fold_size
        
        val_paths = paths[start_idx:end_idx]
        train_paths = paths[:start_idx] + paths[end_idx:]
        
        yield train_paths, val_paths
        
        start_idx = end_idx

def train_kfold(yaml_configuration_file:Path|str, k:int):
    LOGGER = logger()

    clean_memory()
    config = load_config(yaml_configuration_file)
    dataset_path = Path(config['data']).parent
    samples = load_samples(dataset_path)
    name = config['name']
    for fold, (train_set, val_set) in enumerate(get_kfold_splits(samples, k=k, shuffle=True, seed=42), start=1):
        LOGGER.info(f"--- Fold {fold} ---")
        LOGGER.debug(f"Train ({len(train_set)})")
        LOGGER.debug(f"Val   ({len(val_set)})")

        samples_to_file(train_set, dataset_path/'train.txt')
        samples_to_file(val_set, dataset_path/'validation.txt')
        config['name'] = name + f' Fold{fold}'

        model = YOLO(config['model'])
        LOGGER.info('Start training')
        train_metrics = model.train(**config)
        LOGGER.info('Training complete')


def train_random_n(yaml_configuration_file:Path|str, n_runs:int, n_samples:int):
    LOGGER = logger()
    clean_memory()

    config = load_config(yaml_configuration_file)
    dataset_path = Path(config['data']).parent
    samples = load_samples(dataset_path)

    name = config['name']
    for i in range(n_runs):
        LOGGER.info(f"--- Run {i+1} ---")
        random.shuffle(samples)
        subsamples = samples[:n_samples]
        split_index = int(len(subsamples)*0.8)
        train_set = subsamples[:split_index]
        val_set = subsamples[split_index:]
        samples_to_file(train_set, dataset_path/'train.txt')
        samples_to_file(val_set, dataset_path/'validation.txt')
        config['name'] = name + f' Run{i+1}'

        model = YOLO(config['model'])
        LOGGER.info('Start training')
        train_metrics = model.train(**config)
        LOGGER.info('Training complete')


def train(yaml_configuration_file:Path|str)->tuple:
    LOGGER = logger()

    clean_memory()
    config = load_config(yaml_configuration_file)

    model = YOLO(config['model'])
    LOGGER.info('Start training')
    train_metrics = model.train(**config)
    LOGGER.info('Training complete')
    result_folder = Path(config['project'])
    
    training_results = (result_folder,) if train_metrics is None else (result_folder, train_metrics)
    return training_results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model using the provided parameters.")
    parser.add_argument('hyp', type=Path, default=None, help='Yaml file containing relevant training parameters. If provided all other command-line arguments will be ignored.')
    parser.add_argument('--folds', type=int, default=None, required=False, help='If provided, enable k-fold cross validation with the provided number of folds.')
    args = parser.parse_args()

    if args.folds is None:
        train(args.hyp)
    else:
        train_kfold(args.hyp, args.folds)
