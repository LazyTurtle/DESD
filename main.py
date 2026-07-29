import yaml, traceback

from pathlib import Path

import train, evaluation

from utils.training import my_logging
from utils.training import shutdown
from utils.training import aggregate_training_results
from utils.training import aggregate_map
from utils.plotting import plot_results_kfold
from utils.plotting import plot_labels

from utils.dataset import subsample

FOLDS = 5
PUBLIC_DATASETS_HYP = [
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

PUBLIC_DATASETS = [
    r"C:\Users\Emys\Pictures\apples\ACFR",
    r"C:\Users\Emys\Pictures\apples\Agroscope Apple",
    r"C:\Users\Emys\Pictures\apples\APPLE MOTS",
    r"C:\Users\Emys\Pictures\apples\Deep Fruits",
    r"C:\Users\Emys\Pictures\apples\Kfuji",
    r"C:\Users\Emys\Pictures\apples\MetaFruit",
    r"C:\Users\Emys\Pictures\apples\MinneApple",
    r"C:\Users\Emys\Pictures\apples\Open Access RGBD",
    r"C:\Users\Emys\Pictures\apples\SMA",
    r"C:\Users\Emys\Pictures\apples\WSU",
]

OPEN_IOT_DATASET = r"C:\Users\Emys\Pictures\OpenIoT"
OPEN_IOT_EVALUATION_DATA = r"C:\Users\Emys\Pictures\OpenIoT\data.yaml"

PD_EVALUATION_FOLDER =                 Path(r'data\public\evaluation')
PD_AGGREGATED_TRAINING_RESULTS =       Path(r'data\public\folds_training')
PD_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\public\folds_evaluation')

PD_PLOT_TRAINING_OUTPUT =              Path(r'data\public\plots_train')
PD_PLOT_EVALUATION_OUTPUT =            Path(r'data\public\plots_eval')
PD_PLOT_LABELS_ANALYSIS =              Path(r'data\public\plots_labels')

SYNTH_A = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic A")
SYNTH_B = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic B")
SYNTH_C = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic C")

SYNTH_AS = Path(r"C:\Users\Emys\Pictures\SyntAC\SA")
SYNTH_BS = Path(r"C:\Users\Emys\Pictures\SyntAC\SB")
SYNTH_CS = Path(r"C:\Users\Emys\Pictures\SyntAC\SC")

HYP_SA =[
    r'config\hyp\SA0.yaml',
    r'config\hyp\SA1.yaml',
    r'config\hyp\SA2.yaml',
    r'config\hyp\SA3.yaml',
    r'config\hyp\SA4.yaml',
]

HYP_SB =[
    r'config\hyp\SB0.yaml',
    r'config\hyp\SB1.yaml',
    r'config\hyp\SB2.yaml',
    r'config\hyp\SB3.yaml',
    r'config\hyp\SB4.yaml',
]

HYP_SC =[
    r'config\hyp\SC0.yaml',
    r'config\hyp\SC1.yaml',
    r'config\hyp\SC2.yaml',
    r'config\hyp\SC3.yaml',
    r'config\hyp\SC4.yaml',
]

SD_PLOT_LABELS_ANALYSIS =               Path(r'data\syn\plots_labels')

SDA_EVALUATION_FOLDER =                 Path(r'data\synA\evaluation')
SDA_AGGREGATED_TRAINING_RESULTS =       Path(r'data\synA\folds_training')
SDA_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\synA\folds_evaluation')
SDA_PLOT_TRAINING_OUTPUT =              Path(r'data\synA\plots_train')
SDA_PLOT_EVALUATION_OUTPUT =            Path(r'data\synA\plots_eval')


SDB_EVALUATION_FOLDER =                 Path(r'data\synB\evaluation')
SDB_AGGREGATED_TRAINING_RESULTS =       Path(r'data\synB\folds_training')
SDB_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\synB\folds_evaluation')
SDB_PLOT_TRAINING_OUTPUT =              Path(r'data\synB\plots_train')
SDB_PLOT_EVALUATION_OUTPUT =            Path(r'data\synB\plots_eval')


SDC_EVALUATION_FOLDER =                 Path(r'data\synC\evaluation')
SDC_AGGREGATED_TRAINING_RESULTS =       Path(r'data\synC\folds_training')
SDC_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\synC\folds_evaluation')
SDC_PLOT_TRAINING_OUTPUT =              Path(r'data\synC\plots_train')
SDC_PLOT_EVALUATION_OUTPUT =            Path(r'data\synC\plots_eval')

def load_yaml(yaml_configuration_file:str|Path):
    yaml_configuration_file = Path(yaml_configuration_file)
    with open(yaml_configuration_file, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

def analyze_datasets(datasets:list[str]|list[Path], out_folder:str|Path):
    out_folder = Path(out_folder)
    plot_labels.plot_number_of_images_per_dataset(datasets, out_folder=out_folder)
    plot_labels.plot_instances_per_image(datasets, out_folder=out_folder)
    plot_labels.plot_sizes(datasets, out_folder=out_folder)
    plot_labels.plot_labels_distribution(datasets, out_folder=out_folder)
    plot_labels.plot_instances(datasets, out_folder=out_folder)

def trainX(
        hs:list, 
        aggregated_result_folder:Path, 
        evaluation_folder:Path, 
        aggregated_evaluation_folder:Path,
        plot_training_folder:Path,
        plot_evaluation_folder:Path
):
    for h in hs:
        config = load_yaml(h)
        folds_root = Path(config['project'])

        train.train_kfold(h, FOLDS)

        aggregate_training_results.aggregate_fold_results(
            folds_root=folds_root, 
            output_folder=aggregated_result_folder / folds_root.stem,
            target_filename='results.csv',
        )

        evaluation.evaluate_folder(
            models_folder=folds_root,
            evaluation_config=OPEN_IOT_EVALUATION_DATA,
            output_folder=evaluation_folder / folds_root.stem,
            only_best=True,
            split='test'
        )

        aggregate_training_results.aggregate_fold_results(
            folds_root=evaluation_folder / folds_root.stem, 
            output_folder=aggregated_evaluation_folder / folds_root.stem,
        )

        aggregate_map.aggregate_map_kfold(
            folds_root=evaluation_folder / folds_root.stem, 
            output_folder=aggregated_evaluation_folder / folds_root.stem,
        )
    plot_results_kfold.plot(aggregated_result_folder, plot_training_folder)
    plot_results_kfold.plot(aggregated_evaluation_folder, plot_evaluation_folder)

def Experiment1():
    # with_open_iot = PUBLIC_DATASETS.copy()
    # with_open_iot.append(OPEN_IOT_DATASET)
    # analyze_datasets(with_open_iot, PD_PLOT_LABELS_ANALYSIS)
    analyze_datasets(PUBLIC_DATASETS, PD_PLOT_LABELS_ANALYSIS)

    trainX(
        PUBLIC_DATASETS_HYP,
        PD_AGGREGATED_TRAINING_RESULTS,
        PD_EVALUATION_FOLDER,
        PD_AGGREGATED_EVALUATION_RESULTS,
        PD_PLOT_TRAINING_OUTPUT,
        PD_PLOT_EVALUATION_OUTPUT
    )

def Experiment2():
    def subsample_datasets():
        subsample.subsample_dataset(SYNTH_A, SYNTH_AS / 'SA0', 1)
        subsample.subsample_dataset(SYNTH_A, SYNTH_AS / 'SA1', 2)
        subsample.subsample_dataset(SYNTH_A, SYNTH_AS / 'SA2', 4)
        subsample.subsample_dataset(SYNTH_A, SYNTH_AS / 'SA3', 8)
        subsample.subsample_dataset(SYNTH_A, SYNTH_AS / 'SA4', 16)

        subsample.subsample_dataset(SYNTH_B, SYNTH_BS / 'SB0', 1)
        subsample.subsample_dataset(SYNTH_B, SYNTH_BS / 'SB1', 2)
        subsample.subsample_dataset(SYNTH_B, SYNTH_BS / 'SB2', 4)
        subsample.subsample_dataset(SYNTH_B, SYNTH_BS / 'SB3', 8)
        subsample.subsample_dataset(SYNTH_B, SYNTH_BS / 'SB4', 16)

        subsample.subsample_dataset(SYNTH_C, SYNTH_CS / 'SC0', 1)
        subsample.subsample_dataset(SYNTH_C, SYNTH_CS / 'SC1', 2)
        subsample.subsample_dataset(SYNTH_C, SYNTH_CS / 'SC2', 4)
        subsample.subsample_dataset(SYNTH_C, SYNTH_CS / 'SC3', 8)
        subsample.subsample_dataset(SYNTH_C, SYNTH_CS / 'SC4', 16)


    # analyze_datasets([SYNTH_A, SYNTH_B, SYNTH_C, Path(OPEN_IOT_DATASET)], SD_PLOT_LABELS_ANALYSIS)
    analyze_datasets([SYNTH_A, SYNTH_B, SYNTH_C], SD_PLOT_LABELS_ANALYSIS)

    # subsample_datasets()

    trainX(
        HYP_SA,
        SDA_AGGREGATED_TRAINING_RESULTS,
        SDA_EVALUATION_FOLDER,
        SDA_AGGREGATED_EVALUATION_RESULTS,
        SDA_PLOT_TRAINING_OUTPUT,
        SDA_PLOT_EVALUATION_OUTPUT
    )

    trainX(
        HYP_SB,
        SDB_AGGREGATED_TRAINING_RESULTS,
        SDB_EVALUATION_FOLDER,
        SDB_AGGREGATED_EVALUATION_RESULTS,
        SDB_PLOT_TRAINING_OUTPUT,
        SDB_PLOT_EVALUATION_OUTPUT
    )

    trainX(
        HYP_SC,
        SDC_AGGREGATED_TRAINING_RESULTS,
        SDC_EVALUATION_FOLDER,
        SDC_AGGREGATED_EVALUATION_RESULTS,
        SDC_PLOT_TRAINING_OUTPUT,
        SDC_PLOT_EVALUATION_OUTPUT
    )


def Experiment3():
    def reduce_dataset():
        minne = Path(r"C:\Users\Emys\Pictures\apples\MinneApple")
        output = Path(r"data\EXPERIMENT3\minne")
        subsample.reduce_dataset(minne, output / 'M0', int(3840 / 4))
        subsample.reduce_dataset(minne, output / 'M1', int(1920 / 4))
        subsample.reduce_dataset(minne, output / 'M2', int(960 / 4))
        subsample.reduce_dataset(minne, output / 'M3', int(480 / 4))
        subsample.reduce_dataset(minne, output / 'M4', int(240 / 4))
    
    reduce_dataset()
    # after this, merge the datasets with a copy of the synthetic datasets of the corresponding size tier
    return

def Experiment4():
    def reduce_dataset():
        minne = Path(r"C:\Users\Emys\Pictures\apples\MinneApple")
        output = Path(r"data\EXPERIMENT4\minne")
        subsample.reduce_dataset(minne, output / 'M0', round(1920 * 0.50))
        subsample.reduce_dataset(minne, output / 'M1', round(1920 * 0.25))
        subsample.reduce_dataset(minne, output / 'M2', round(1920 * 0.10))
        subsample.reduce_dataset(minne, output / 'M3', round(1920 * 0.05))
        subsample.reduce_dataset(minne, output / 'M4', round(1920 * 0.02))
    
    reduce_dataset()
    # after this, merge the datasets with copies of SB1
    return

if __name__ == "__main__":
    logger = my_logging.get_logger('Train On Datasets', out_folder='logs/train_on_datasets')

    try:
        Experiment1()
        Experiment2()

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