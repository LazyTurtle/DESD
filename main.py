import yaml, traceback

from pathlib import Path

import train, evaluation

from utils.training import my_logging
from utils.training import shutdown
from utils.training import aggregate_training_results
from utils.plotting import plot_results_kfold

FOLDS = 5
HYP = [
    r"config\hyp\acfr.yaml",
    r"config\hyp\agroscope.yaml",
    r"config\hyp\apple_mots.yaml",
    r"config\hyp\deep_fruits.yaml",
    r"config\hyp\kfuji.yaml",
    r"config\hyp\meta_fruit.yaml",
    r"config\hyp\minne_apple.yaml",
    r"config\hyp\open.yaml",
    r"config\hyp\sma.yaml",
    r"config\hyp\wsu.yaml",
]

EVALUATION_FOLDER =                 Path(r'data\evaluation')
AGGREGATED_TRAINING_RESULTS =       Path(r'data\folds_training')
AGGREGATED_EVALUATION_RESULTS =     Path(r'data\folds_evaluation')
OPEN_IOT_DATASET =                  Path(r"C:\Users\Emys\Pictures\massimo\data.yaml")

PLOT_TRAINING_OUTPUT =              Path(r'data\plots_train')
PLOT_EVALUATION_OUTPUT =            Path(r'data\plots_eval')

def load_yaml(yaml_configuration_file:str|Path):
    yaml_configuration_file = Path(yaml_configuration_file)
    with open(yaml_configuration_file, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    logger = my_logging.get_logger('Train On Datasets', out_folder='logs/train_on_datasets')
    try:

        for h in HYP:
            train.train_kfold(h, FOLDS)

            config = load_yaml(h)
            folds_root = Path(config['project'])
            aggregate_training_results.aggregate_fold_results(
                folds_root=folds_root, 
                output_folder=AGGREGATED_TRAINING_RESULTS / folds_root.stem,
                target_filename='results.csv',
            )

            evaluation.evaluate_folder(
                models_folder=folds_root,
                evaluation_config=OPEN_IOT_DATASET,
                output_folder=EVALUATION_FOLDER / folds_root.stem,
                only_best=True,
                split='test'
            )

            aggregate_training_results.aggregate_fold_results(
                folds_root=EVALUATION_FOLDER / folds_root.stem, 
                output_folder=AGGREGATED_EVALUATION_RESULTS / folds_root.stem,
            )

        plot_results_kfold.plot(AGGREGATED_TRAINING_RESULTS, PLOT_TRAINING_OUTPUT)
        plot_results_kfold.plot(AGGREGATED_EVALUATION_RESULTS, PLOT_EVALUATION_OUTPUT)

        
            
    except KeyboardInterrupt:
        logger.info("User interrupted the program")
    except Exception as e:
        logger.error("Unforeseen error")
        logger.error(e)
        logger.error(traceback.format_exc())
    except BaseException as e:
        logger.error("Catastrophic error")
        logger.error(e)
        logger.error(traceback.format_exc())
    finally:
        logger.info("Script complete")

    timer = shutdown.ShutdownTimer()
    timer.start()