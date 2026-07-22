import os, csv, argparse, numpy as np

from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

def __get_models(models_folder:Path, only_best:bool=False) -> list[Path]:
    model_list = list()
    pattern = "**/*.pt" if not only_best else "**/best.pt"
    model_paths = models_folder.glob(pattern, recurse_symlinks=True)
    for model_path in model_paths:
        model_list.append(Path(model_path))
    return model_list

def __save_metrics(metrics:DetMetrics, folder:Path) -> list[Path]:
    curves_data = metrics.curves_results

    confidence = list(curves_data[1][0].flatten())
    f1_curve = np.mean(curves_data[1][1], axis=0)
    precision_curve = np.mean(curves_data[2][1], axis=0)
    recall_curve = np.mean(curves_data[3][1], axis=0)
    confidence_graphs = {
        'Confidence' : confidence,
        'F1 Score' : f1_curve,
        'Precision' : precision_curve,
        'Recall' : recall_curve
    }
    __save_to_csv(folder/'detection_results.csv', confidence_graphs)

    precision_curve = np.mean(curves_data[0][1], axis=0)
    precision_recall = {
    'PR Recall' : list(curves_data[0][0].flatten()),
    'PR' : list(precision_curve.flatten()),
    }
    __save_to_csv(folder/'precision_recall.csv', precision_recall)

    map = {
    'Confidence' : [50, 95],
    'mAP' : [metrics.box.map50, metrics.box.map]
    }
    __save_to_csv(folder/'map.csv', map)

    files = [
    folder/'detection_results.csv',
    folder/'precision_recall.csv',
    folder/'map.csv',
    ]
    
    return files

def __save_to_csv(file_name:Path, dictionary:dict):
  
  rows = zip(*list(dictionary.values()))
  with open(file_name,'w') as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(list(dictionary.keys()))
    for row in rows:
      writer.writerow(row)

def evaluate_model(model_path:Path|str, evaluation_config:Path|str, output_folder:str|Path, image_size:int=640, single_class:bool=False, name:str|None=None, split:str='val')->list[Path]:
    model_path = Path(model_path)
    evaluation_config = Path(evaluation_config)
    output_folder = Path(output_folder)
    home = Path(os.getcwd().strip())

    model = YOLO(model_path)
    # the subfolder name should be the name of the run, YOLO saves the model into path/to/RUN_NAME/weights/best.pt
    subfolder_mane = name if name is not None else f'{str(model_path.parent.parent.stem)}'

    arguments = {
        "data" : str(evaluation_config),
        "plots" : True,
        "project" : home.joinpath(output_folder),
        "name" : subfolder_mane,
        "single_cls" : single_class,
        "imgsz": image_size,
        "save_txt": True,
        'split':split
    }
    metrics = model.val(**arguments)

    metric_folder = output_folder.joinpath(subfolder_mane).joinpath('evaluation_files')
    metric_folder.mkdir(parents=True, exist_ok=True)
    files = __save_metrics(metrics, metric_folder)
    return files

def evaluate_folder(models_folder:str|Path, evaluation_config:str|Path, output_folder:str|Path, image_size:int=640, only_best:bool=False, single_class:bool=False, split:str='val'):
    models_folder = Path(models_folder)
    evaluation_config = Path(evaluation_config)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    
    model_paths = __get_models(models_folder, only_best)
    print('Found the following models:')
    print(str(model_paths))

    for model_path in model_paths:
        evaluate_model(
           model_path=model_path,
           evaluation_config=evaluation_config,
           output_folder=output_folder,
           image_size=image_size,
           single_class=single_class,
           name = model_path.stem if not only_best else None,
           split = split
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the models present in a directory with the provided data.")
    parser.add_argument("models_folder", type=Path, help="Path to the folder containing YOLO weights `.pt` files.")
    parser.add_argument("evaluation_config", type=Path, help='Path to the yaml config file used for evaluation.')
    parser.add_argument("output", type=Path, help="Directory to save the output plot.")
    parser.add_argument("--image-size", required=False, type=int, default=640, help="Size of the image to apply during inference.")
    parser.add_argument("--only-best", required=False, action=argparse.BooleanOptionalAction, default=False, help="Use only the 'best' models provided by ultralytics training.")
    parser.add_argument("--single-class", required=False, action=argparse.BooleanOptionalAction, default=False, help="Use the single-class mode for the detectors.")
    parser.add_argument("--split", choices=['val', 'test'], default='val', type=str, help="Validation set.")

    args = parser.parse_args()

    out_folder = args.output if args.output is not None else args.result_folder
    evaluate_folder(
        models_folder = args.models_folder,
        evaluation_config = args.evaluation_config,
        output_folder = out_folder,
        image_size = args.image_size,
        only_best = args.only_best,
        single_class = args.single_class,
        split = args.split,
    )

    