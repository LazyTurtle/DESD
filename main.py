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
from utils.dataset import duplicates

FOLDS = 5

################################################################################
#################### E1
################################################################################

PUBLIC_DATASETS_HYP = [
    r"config\hyp\E1\acfr.yaml",
    r"config\hyp\E1\agroscope.yaml",
    r"config\hyp\E1\apple_mots.yaml",
    r"config\hyp\E1\deep_fruits.yaml",
    r"config\hyp\E1\kfuji.yaml",
    r"config\hyp\E1\meta_fruit.yaml",
    r"config\hyp\E1\minne_apple.yaml",
    r"config\hyp\E1\open.yaml",
    r"config\hyp\E1\sma.yaml",
    r"config\hyp\E1\wsu.yaml",
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

PD_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT1\evaluation')
PD_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT1\folds_training')
PD_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT1\folds_evaluation')
PD_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT1\plots_train')
PD_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT1\plots_eval')
PD_PLOT_LABELS_ANALYSIS =              Path(r'data\EXPERIMENT1\plots_labels')

################################################################################
#################### E2
################################################################################

HYP_SA =[
    r'config\hyp\E2\SA0.yaml',
    r'config\hyp\E2\SA1.yaml',
    r'config\hyp\E2\SA2.yaml',
    r'config\hyp\E2\SA3.yaml',
    r'config\hyp\E2\SA4.yaml',
]

HYP_SB =[
    r'config\hyp\E2\SB0.yaml',
    r'config\hyp\E2\SB1.yaml',
    r'config\hyp\E2\SB2.yaml',
    r'config\hyp\E2\SB3.yaml',
    r'config\hyp\E2\SB4.yaml',
]

HYP_SC =[
    r'config\hyp\E2\SC0.yaml',
    r'config\hyp\E2\SC1.yaml',
    r'config\hyp\E2\SC2.yaml',
    r'config\hyp\E2\SC3.yaml',
    r'config\hyp\E2\SC4.yaml',
]

SD_PLOT_LABELS_ANALYSIS =               Path(r'data\EXPERIMENT2\plots_labels')

SDA_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT2\A\evaluation')
SDA_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT2\A\folds_training')
SDA_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT2\A\folds_evaluation')
SDA_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT2\A\plots_train')
SDA_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT2\A\plots_eval')


SDB_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT2\B\evaluation')
SDB_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT2\B\folds_training')
SDB_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT2\B\folds_evaluation')
SDB_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT2\B\plots_train')
SDB_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT2\B\plots_eval')


SDC_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT2\C\evaluation')
SDC_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT2\C\folds_training')
SDC_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT2\C\folds_evaluation')
SDC_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT2\C\plots_train')
SDC_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT2\C\plots_eval')

################################################################################
#################### E2b
################################################################################

SDA_B_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT2B\A\evaluation')
SDA_B_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT2B\A\folds_training')
SDA_B_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT2B\A\folds_evaluation')
SDA_B_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT2B\A\plots_train')
SDA_B_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT2B\A\plots_eval')


SDB_B_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT2B\B\evaluation')
SDB_B_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT2B\B\folds_training')
SDB_B_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT2B\B\folds_evaluation')
SDB_B_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT2B\B\plots_train')
SDB_B_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT2B\B\plots_eval')


SDC_B_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT2B\C\evaluation')
SDC_B_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT2B\C\folds_training')
SDC_B_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT2B\C\folds_evaluation')
SDC_B_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT2B\C\plots_train')
SDC_B_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT2B\C\plots_eval')

################################################################################
#################### E3
################################################################################

HYP_SMA =[
    r'config\hyp\E3\SMA0.yaml',
    r'config\hyp\E3\SMA1.yaml',
    r'config\hyp\E3\SMA2.yaml',
    r'config\hyp\E3\SMA3.yaml',
    r'config\hyp\E3\SMA4.yaml',
]

HYP_SMB =[
    r'config\hyp\E3\SMB0.yaml',
    r'config\hyp\E3\SMB1.yaml',
    r'config\hyp\E3\SMB2.yaml',
    r'config\hyp\E3\SMB3.yaml',
    r'config\hyp\E3\SMB4.yaml',
]

HYP_SMC =[
    r'config\hyp\E3\SMC0.yaml',
    r'config\hyp\E3\SMC1.yaml',
    r'config\hyp\E3\SMC2.yaml',
    r'config\hyp\E3\SMC3.yaml',
    r'config\hyp\E3\SMC4.yaml',
]

SMA_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT3\A\evaluation')
SMA_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT3\A\folds_training')
SMA_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT3\A\folds_evaluation')
SMA_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT3\A\plots_train')
SMA_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT3\A\plots_eval')


SMB_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT3\B\evaluation')
SMB_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT3\B\folds_training')
SMB_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT3\B\folds_evaluation')
SMB_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT3\B\plots_train')
SMB_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT3\B\plots_eval')


SMC_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT3\C\evaluation')
SMC_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT3\C\folds_training')
SMC_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT3\C\folds_evaluation')
SMC_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT3\C\plots_train')
SMC_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT3\C\plots_eval')


################################################################################
#################### E4
################################################################################
HYP_SN =[
    r'config\hyp\E4\SN00.yaml',
    r'config\hyp\E4\SN02.yaml',
    r'config\hyp\E4\SN05.yaml',
    r'config\hyp\E4\SN10.yaml',
    r'config\hyp\E4\SN25.yaml',
    r'config\hyp\E4\SN50.yaml',
]

SN_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT4\SN\evaluation')
SN_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT4\SN\folds_training')
SN_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT4\SN\folds_evaluation')
SN_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT4\SN\plots_train')
SN_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT4\SN\plots_eval')

HYP_SF =[
    r'config\hyp\E4\SF00.yaml',
    r'config\hyp\E4\SF02.yaml',
    r'config\hyp\E4\SF05.yaml',
    r'config\hyp\E4\SF10.yaml',
    r'config\hyp\E4\SF25.yaml',
    r'config\hyp\E4\SF50.yaml',
]

SF_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT4\SF\evaluation')
SF_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT4\SF\folds_training')
SF_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT4\SF\folds_evaluation')
SF_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT4\SF\plots_train')
SF_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT4\SF\plots_eval')

HYP_SM =[
    r'config\hyp\E4\SM00.yaml',
    r'config\hyp\E4\SM02.yaml',
    r'config\hyp\E4\SM05.yaml',
    r'config\hyp\E4\SM10.yaml',
    r'config\hyp\E4\SM25.yaml',
    r'config\hyp\E4\SM50.yaml',
]

SM_EVALUATION_FOLDER =                 Path(r'data\EXPERIMENT4\SM\evaluation')
SM_AGGREGATED_TRAINING_RESULTS =       Path(r'data\EXPERIMENT4\SM\folds_training')
SM_AGGREGATED_EVALUATION_RESULTS =     Path(r'data\EXPERIMENT4\SM\folds_evaluation')
SM_PLOT_TRAINING_OUTPUT =              Path(r'data\EXPERIMENT4\SM\plots_train')
SM_PLOT_EVALUATION_OUTPUT =            Path(r'data\EXPERIMENT4\SM\plots_eval')


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


def train_random_subsample(
        h:str, 
        aggregated_result_folder:Path, 
        evaluation_folder:Path, 
        aggregated_evaluation_folder:Path,
        samples:int,
):
    config = load_yaml(h)
    folds_root = Path(config['project'])

    train.train_random_n(h, 5, samples)

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

def Experiment1():
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
    A = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic A")
    B = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic B")
    C = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic C")
    def produce_datasets():

        SA = Path(r"data\EXPERIMENT2\SA")
        SB = Path(r"data\EXPERIMENT2\SB")
        SC = Path(r"data\EXPERIMENT2\SC")

        subsample.subsample_dataset(A, SA / 'SA0', 1)
        subsample.subsample_dataset(A, SA / 'SA1', 2)
        subsample.subsample_dataset(A, SA / 'SA2', 4)
        subsample.subsample_dataset(A, SA / 'SA3', 8)
        subsample.subsample_dataset(A, SA / 'SA4', 16)

        subsample.subsample_dataset(B, SB / 'SB0', 1)
        subsample.subsample_dataset(B, SB / 'SB1', 2)
        subsample.subsample_dataset(B, SB / 'SB2', 4)
        subsample.subsample_dataset(B, SB / 'SB3', 8)
        subsample.subsample_dataset(B, SB / 'SB4', 16)

        subsample.subsample_dataset(C, SC / 'SC0', 1)
        subsample.subsample_dataset(C, SC / 'SC1', 2)
        subsample.subsample_dataset(C, SC / 'SC2', 4)
        subsample.subsample_dataset(C, SC / 'SC3', 8)
        subsample.subsample_dataset(C, SC / 'SC4', 16)


    analyze_datasets([A, B, C], SD_PLOT_LABELS_ANALYSIS)

    produce_datasets()

    # EXPERIMENT E2a with controlled downsampling of the synthetic datasets
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

    # EXPERIMENT E2b with random downsampling of the synthetic datasets

    train_random_subsample(
        r'config\hyp\E2b\SA1.yaml',
        SDA_B_AGGREGATED_TRAINING_RESULTS,
        SDA_B_EVALUATION_FOLDER,
        SDA_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/2),
    )
    train_random_subsample(
        r'config\hyp\E2b\SA2.yaml',
        SDA_B_AGGREGATED_TRAINING_RESULTS,
        SDA_B_EVALUATION_FOLDER,
        SDA_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/4),
    )
    train_random_subsample(
        r'config\hyp\E2b\SA3.yaml',
        SDA_B_AGGREGATED_TRAINING_RESULTS,
        SDA_B_EVALUATION_FOLDER,
        SDA_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/8),
    )
    train_random_subsample(
        r'config\hyp\E2b\SA4.yaml',
        SDA_B_AGGREGATED_TRAINING_RESULTS,
        SDA_B_EVALUATION_FOLDER,
        SDA_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/16),
    )
    plot_results_kfold.plot(SDA_B_AGGREGATED_TRAINING_RESULTS, SDA_B_PLOT_TRAINING_OUTPUT)
    plot_results_kfold.plot(SDA_B_AGGREGATED_EVALUATION_RESULTS, SDA_B_PLOT_EVALUATION_OUTPUT)
    
    train_random_subsample(
        r'config\hyp\E2b\SB1.yaml',
        SDB_B_AGGREGATED_TRAINING_RESULTS,
        SDB_B_EVALUATION_FOLDER,
        SDB_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/2),
    )
    train_random_subsample(
        r'config\hyp\E2b\SB2.yaml',
        SDB_B_AGGREGATED_TRAINING_RESULTS,
        SDB_B_EVALUATION_FOLDER,
        SDB_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/4),
    )
    train_random_subsample(
        r'config\hyp\E2b\SB3.yaml',
        SDB_B_AGGREGATED_TRAINING_RESULTS,
        SDB_B_EVALUATION_FOLDER,
        SDB_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/8),
    )
    train_random_subsample(
        r'config\hyp\E2b\SB4.yaml',
        SDB_B_AGGREGATED_TRAINING_RESULTS,
        SDB_B_EVALUATION_FOLDER,
        SDB_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/16),
    )
    plot_results_kfold.plot(SDB_B_AGGREGATED_TRAINING_RESULTS, SDB_B_PLOT_TRAINING_OUTPUT)
    plot_results_kfold.plot(SDB_B_AGGREGATED_EVALUATION_RESULTS, SDB_B_PLOT_EVALUATION_OUTPUT)

    train_random_subsample(
        r'config\hyp\E2b\SC1.yaml',
        SDC_B_AGGREGATED_TRAINING_RESULTS,
        SDC_B_EVALUATION_FOLDER,
        SDC_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/2),
    )
    train_random_subsample(
        r'config\hyp\E2b\SC2.yaml',
        SDC_B_AGGREGATED_TRAINING_RESULTS,
        SDC_B_EVALUATION_FOLDER,
        SDC_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/4),
    )
    train_random_subsample(
        r'config\hyp\E2b\SC3.yaml',
        SDC_B_AGGREGATED_TRAINING_RESULTS,
        SDC_B_EVALUATION_FOLDER,
        SDC_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/8),
    )
    train_random_subsample(
        r'config\hyp\E2b\SC4.yaml',
        SDC_B_AGGREGATED_TRAINING_RESULTS,
        SDC_B_EVALUATION_FOLDER,
        SDC_B_AGGREGATED_EVALUATION_RESULTS,
        int(3840/16),
    )
    plot_results_kfold.plot(SDC_B_AGGREGATED_TRAINING_RESULTS, SDC_B_PLOT_TRAINING_OUTPUT)
    plot_results_kfold.plot(SDC_B_AGGREGATED_EVALUATION_RESULTS, SDC_B_PLOT_EVALUATION_OUTPUT)


def Experiment3():
    def produce_datasets():
        minne = Path(r"C:\Users\Emys\Pictures\apples\MinneApple")
        output = Path(r"data\EXPERIMENT3\minne")
        subsample.reduce_dataset(minne, output / 'M0', int(3840 * (1/5)))
        subsample.reduce_dataset(minne, output / 'M1', int(1920 * (1/5)))
        subsample.reduce_dataset(minne, output / 'M2', int(960 * (1/5)))
        subsample.reduce_dataset(minne, output / 'M3', int(480 * (1/5)))
        subsample.reduce_dataset(minne, output / 'M4', int(240 * (1/5)))

        A = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic A")
        A_out = Path(r"data\EXPERIMENT3\OA")
        subsample.reduce_dataset(A, A_out / 'A0', int(3840 * (4/5)))
        subsample.reduce_dataset(A, A_out / 'A1', int(1920 * (4/5)))
        subsample.reduce_dataset(A, A_out / 'A2', int(960 * (4/5)))
        subsample.reduce_dataset(A, A_out / 'A3', int(480 * (4/5)))
        subsample.reduce_dataset(A, A_out / 'A4', int(240 * (4/5)))

        B = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic B")
        B_out = Path(r"data\EXPERIMENT3\OB")
        subsample.reduce_dataset(B, B_out / 'B0', int(3840 * (4/5)))
        subsample.reduce_dataset(B, B_out / 'B1', int(1920 * (4/5)))
        subsample.reduce_dataset(B, B_out / 'B2', int(960 * (4/5)))
        subsample.reduce_dataset(B, B_out / 'B3', int(480 * (4/5)))
        subsample.reduce_dataset(B, B_out / 'B4', int(240 * (4/5)))

        C = Path(r"C:\Users\Emys\Pictures\SyntAC\Synthetic C")
        C_out = Path(r"data\EXPERIMENT3\OC")
        subsample.reduce_dataset(C, C_out / 'C0', int(3840 * (4/5)))
        subsample.reduce_dataset(C, C_out / 'C1', int(1920 * (4/5)))
        subsample.reduce_dataset(C, C_out / 'C2', int(960 * (4/5)))
        subsample.reduce_dataset(C, C_out / 'C3', int(480 * (4/5)))
        subsample.reduce_dataset(C, C_out / 'C4', int(240 * (4/5)))

    
    produce_datasets()
    # after this, merge the reduced minne dataset with the reduced copies of the synthetic datasets of the corresponding size tier
    return

    trainX(
        HYP_SMA,
        SMA_AGGREGATED_TRAINING_RESULTS,
        SMA_EVALUATION_FOLDER,
        SMA_AGGREGATED_EVALUATION_RESULTS,
        SMA_PLOT_TRAINING_OUTPUT,
        SMA_PLOT_EVALUATION_OUTPUT
    )

    trainX(
        HYP_SMB,
        SMB_AGGREGATED_TRAINING_RESULTS,
        SMB_EVALUATION_FOLDER,
        SMB_AGGREGATED_EVALUATION_RESULTS,
        SMB_PLOT_TRAINING_OUTPUT,
        SMB_PLOT_EVALUATION_OUTPUT
    )

    trainX(
        HYP_SMC,
        SMC_AGGREGATED_TRAINING_RESULTS,
        SMC_EVALUATION_FOLDER,
        SMC_AGGREGATED_EVALUATION_RESULTS,
        SMC_PLOT_TRAINING_OUTPUT,
        SMC_PLOT_EVALUATION_OUTPUT
    )


def Experiment4():
    def produce_datasets():
        dataset = Path(r"C:\Users\Emys\Pictures\SyntAC\SC\SC3")
        output = Path(r"data\EXPERIMENT4\C")
        subsample.reduce_dataset(dataset, output / 'C50', round(480 * (1 - 0.50)))
        subsample.reduce_dataset(dataset, output / 'C25', round(480 * (1 - 0.25)))
        subsample.reduce_dataset(dataset, output / 'C10', round(480 * (1 - 0.10)))
        subsample.reduce_dataset(dataset, output / 'C05', round(480 * (1 - 0.05)))
        subsample.reduce_dataset(dataset, output / 'C02', round(480 * (1 - 0.02)))

        dataset = Path(r"C:\Users\Emys\Pictures\apples\MinneApple")
        output = Path(r"data\EXPERIMENT4\minne")
        subsample.reduce_dataset(dataset, output / 'M50', round(480 * 0.50))
        subsample.reduce_dataset(dataset, output / 'M25', round(480 * 0.25))
        subsample.reduce_dataset(dataset, output / 'M10', round(480 * 0.10))
        subsample.reduce_dataset(dataset, output / 'M05', round(480 * 0.05))
        subsample.reduce_dataset(dataset, output / 'M02', round(480 * 0.02))

        dataset = Path(r"C:\Users\Emys\Pictures\apples\MetaFruit")
        output = Path(r"data\EXPERIMENT4\meta_fruit")
        subsample.reduce_dataset(dataset, output / 'F50', round(480 * 0.50))
        subsample.reduce_dataset(dataset, output / 'F25', round(480 * 0.25))
        subsample.reduce_dataset(dataset, output / 'F10', round(480 * 0.10))
        subsample.reduce_dataset(dataset, output / 'F05', round(480 * 0.05))
        subsample.reduce_dataset(dataset, output / 'F02', round(480 * 0.02))

        dataset = Path(r"C:\Users\Emys\Pictures\apples\APPLE MOTS")
        output = Path(r"data\EXPERIMENT4\mots")
        subsample.reduce_dataset(dataset, output / 'N50', round(480 * 0.50))
        subsample.reduce_dataset(dataset, output / 'N25', round(480 * 0.25))
        subsample.reduce_dataset(dataset, output / 'N10', round(480 * 0.10))
        subsample.reduce_dataset(dataset, output / 'N05', round(480 * 0.05))
        subsample.reduce_dataset(dataset, output / 'N02', round(480 * 0.02))
    
    produce_datasets()
    # after this, merge the datasets with the corresponding subsets of SC3
    return

    trainX(
        HYP_SM,
        SM_AGGREGATED_TRAINING_RESULTS,
        SM_EVALUATION_FOLDER,
        SM_AGGREGATED_EVALUATION_RESULTS,
        SM_PLOT_TRAINING_OUTPUT,
        SM_PLOT_EVALUATION_OUTPUT
    )

    trainX(
        HYP_SN,
        SN_AGGREGATED_TRAINING_RESULTS,
        SN_EVALUATION_FOLDER,
        SN_AGGREGATED_EVALUATION_RESULTS,
        SN_PLOT_TRAINING_OUTPUT,
        SN_PLOT_EVALUATION_OUTPUT
    )

    trainX(
        HYP_SF,
        SF_AGGREGATED_TRAINING_RESULTS,
        SF_EVALUATION_FOLDER,
        SF_AGGREGATED_EVALUATION_RESULTS,
        SF_PLOT_TRAINING_OUTPUT,
        SF_PLOT_EVALUATION_OUTPUT
    )

def Experiment5():
    from bootstrap_f1 import bootstrap_f1_difference, _save_output, _print_report
    def bootstrap(A, B, output:Path|str = Path('data/bootstrap'), conf = 0.5):
        A = Path(A)
        B = Path(B)
        output = Path(output)
        results = bootstrap_f1_difference(A, B, OPEN_IOT_DATASET, conf)
        _save_output(results,Path(output / f'{A.stem}-{B.stem}'))
        _print_report(results)

    # E2a
    bootstrap(
        r'data\EXPERIMENT2\C\training\SC\SC3',
        r'data\EXPERIMENT2\A\training\SA\SA3',
        output=r'data/bootstrap/E2a'
    )
    bootstrap(
        r'data\EXPERIMENT2\C\training\SC\SC3',
        r'data\EXPERIMENT2\B\training\SB\SB3',
        output=r'data/bootstrap/E2a'
    )
    bootstrap(
        r'data\EXPERIMENT2\B\training\SB\SB3',
        r'data\EXPERIMENT2\A\training\SA\SA3',
        output=r'data/bootstrap/E2a'
    )
    
    # E2b
    bootstrap(
        r'data\EXPERIMENT2\A\training\SA\SA3',
        r'data\EXPERIMENT2B\A\training\SA\SA3',
        output=r'data/bootstrap/E2b'
    )
    bootstrap(
        r'data\EXPERIMENT2\B\training\SB\SB3',
        r'data\EXPERIMENT2B\B\training\SB\SB3',
        output=r'data/bootstrap/E2b'
    )
    bootstrap(
        r'data\EXPERIMENT2\C\training\SC\SC3',
        r'data\EXPERIMENT2B\C\training\SC\SC3',
        output=r'data/bootstrap/E2b'
    )

    # E3
    bootstrap(
        r'data\EXPERIMENT1\training\PUBLIC_DATASETS\MinneApple',
        r'data\EXPERIMENT3\A\training\SMA\SMA1',
        output=r'data/bootstrap/E3'
    )
    bootstrap(
        r'data\EXPERIMENT1\training\PUBLIC_DATASETS\MinneApple',
        r'data\EXPERIMENT3\B\training\SMB\SMB1',
        output=r'data/bootstrap/E3'
    )
    bootstrap(
        r'data\EXPERIMENT1\training\PUBLIC_DATASETS\MinneApple',
        r'data\EXPERIMENT3\C\training\SMC\SMC1',
        output=r'data/bootstrap/E3'
    )


    # E4
    bootstrap(
        r'data\EXPERIMENT4\SM\training\SM50',
        r'data\EXPERIMENT4\SM\training\SM25',
        output=r'data/bootstrap/E4'
    )
    bootstrap(
        r'data\EXPERIMENT4\SM\training\SM50',
        r'data\EXPERIMENT4\SM\training\SM10',
        output=r'data/bootstrap/E4'
    )
    bootstrap(
        r'data\EXPERIMENT4\SM\training\SM50',
        r'data\EXPERIMENT4\SM\training\SM05',
        output=r'data/bootstrap/E4'
    )
    bootstrap(
        r'data\EXPERIMENT4\SM\training\SM50',
        r'data\EXPERIMENT4\SM\training\SM02',
        output=r'data/bootstrap/E4'
    )
    bootstrap(
        r'data\EXPERIMENT4\SM\training\SM50',
        r'data\EXPERIMENT4\SM\training\SM00',
        output=r'data/bootstrap/E4'
    )
    bootstrap(
        r'data\EXPERIMENT4\SM\training\SM10',
        r'data\EXPERIMENT4\SM\training\SM05',
        output=r'data/bootstrap/E4'
    )

    # E2-E3
    bootstrap(
        r'data\EXPERIMENT2\A\training\SA\SA3',
        r'data\EXPERIMENT3\A\training\SMA\SMA3',
        output=r'data/bootstrap/E2-E3'
    )
    bootstrap(
        r'data\EXPERIMENT2\B\training\SB\SB3',
        r'data\EXPERIMENT3\B\training\SMB\SMB3',
        output=r'data/bootstrap/E2-E3'
    )
    bootstrap(
        r'data\EXPERIMENT2\C\training\SC\SC3',
        r'data\EXPERIMENT3\C\training\SMC\SMC3',
        output=r'data/bootstrap/E2-E3'
    )



def CalculateNearDuplicates():
    logs = ['|Dataset Name|pHash|DINO|']
    datasets = [
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
        r"C:\Users\Emys\Pictures\OpenIoT",
        r"C:\Users\Emys\Pictures\SyntAC\Synthetic C",
        r"C:\Users\Emys\Pictures\SyntAC\Synthetic A",
        r"C:\Users\Emys\Pictures\SyntAC\Synthetic B",
        r"C:\Users\Emys\Pictures\SyntAC\SA\SA0",
        r"C:\Users\Emys\Pictures\SyntAC\SA\SA1",
        r"C:\Users\Emys\Pictures\SyntAC\SA\SA2",
        r"C:\Users\Emys\Pictures\SyntAC\SA\SA3",
        r"C:\Users\Emys\Pictures\SyntAC\SA\SA4",
        r"C:\Users\Emys\Pictures\SyntAC\SB\SB0",
        r"C:\Users\Emys\Pictures\SyntAC\SB\SB1",
        r"C:\Users\Emys\Pictures\SyntAC\SB\SB2",
        r"C:\Users\Emys\Pictures\SyntAC\SB\SB3",
        r"C:\Users\Emys\Pictures\SyntAC\SB\SB4",
        r"C:\Users\Emys\Pictures\SyntAC\SC\SC0",
        r"C:\Users\Emys\Pictures\SyntAC\SC\SC1",
        r"C:\Users\Emys\Pictures\SyntAC\SC\SC2",
        r"C:\Users\Emys\Pictures\SyntAC\SC\SC3",
        r"C:\Users\Emys\Pictures\SyntAC\SC\SC4",
        r"data\EXPERIMENT2\SAb\SAb1",
        r"data\EXPERIMENT2\SAb\SAb2",
        r"data\EXPERIMENT2\SAb\SAb3",
        r"data\EXPERIMENT2\SAb\SAb4",
        r"data\EXPERIMENT2\SBb\SBb1",
        r"data\EXPERIMENT2\SBb\SBb2",
        r"data\EXPERIMENT2\SBb\SBb3",
        r"data\EXPERIMENT2\SBb\SBb4",
        r"data\EXPERIMENT2\SCb\SCb1",
        r"data\EXPERIMENT2\SCb\SCb2",
        r"data\EXPERIMENT2\SCb\SCb3",
        r"data\EXPERIMENT2\SCb\SCb4",
        r"data\EXPERIMENT3\A\SMA0",
        r"data\EXPERIMENT3\A\SMA1",
        r"data\EXPERIMENT3\A\SMA2",
        r"data\EXPERIMENT3\A\SMA3",
        r"data\EXPERIMENT3\A\SMA4",
        r"data\EXPERIMENT3\B\SMB0",
        r"data\EXPERIMENT3\B\SMB1",
        r"data\EXPERIMENT3\B\SMB2",
        r"data\EXPERIMENT3\B\SMB3",
        r"data\EXPERIMENT3\B\SMB4",
        r"data\EXPERIMENT3\C\SMC0",
        r"data\EXPERIMENT3\C\SMC1",
        r"data\EXPERIMENT3\C\SMC2",
        r"data\EXPERIMENT3\C\SMC3",
        r"data\EXPERIMENT3\C\SMC4",
        r"data\EXPERIMENT4\SF\F00",
        r"data\EXPERIMENT4\SF\F02",
        r"data\EXPERIMENT4\SF\F05",
        r"data\EXPERIMENT4\SF\F10",
        r"data\EXPERIMENT4\SF\F25",
        r"data\EXPERIMENT4\SF\F50",
        r"data\EXPERIMENT4\SM\M00",
        r"data\EXPERIMENT4\SM\M02",
        r"data\EXPERIMENT4\SM\M05",
        r"data\EXPERIMENT4\SM\M10",
        r"data\EXPERIMENT4\SM\M25",
        r"data\EXPERIMENT4\SM\M50",
        r"data\EXPERIMENT4\SN\N00",
        r"data\EXPERIMENT4\SN\N02",
        r"data\EXPERIMENT4\SN\N05",
        r"data\EXPERIMENT4\SN\N10",
        r"data\EXPERIMENT4\SN\N25",
        r"data\EXPERIMENT4\SN\N50",
    ]
    for dataset in datasets:
        dataset_name = Path(dataset).stem
        rates = duplicates.compute_near_duplicates(dataset)
        logs.append(f"|{dataset_name}|{rates[0]:.1%}|{rates[1]:.1%}|")
        
    with open('data/near_duplicate_report.md', 'w') as report:
        report.writelines("\n".join(logs))


if __name__ == "__main__":
    logger = my_logging.get_logger('ExperimentPipeline', out_folder='logs/pipeline')

    try:
        pass
        Experiment1()
        logger.info('Experiment 1 complete.')
        Experiment2()
        logger.info('Experiment 2 complete.')
        Experiment3()
        logger.info('Experiment 3 complete.')
        Experiment4()
        logger.info('Experiment 4 complete.')
        Experiment5()
        logger.info('Experiment 5 complete.')

        CalculateNearDuplicates()

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