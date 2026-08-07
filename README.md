# DESD

Code accompanying **"Designing Effective Synthetic Datasets for Fruit Detection in Agriculture"** (Hueller, Vecchio, Antonelli, submitted to *Smart Agricultural Technology*, 2026), which benchmarks YOLO-based apple detection trained on public real-world datasets, procedurally generated synthetic datasets, and real+synthetic hybrid mixes.

## Contents
Training and evaluation scripts (5-fold cross-validation using YOLO). Training hyperparameters (contained in `config\hyp\*`) are set as defaults in `main.py`.

## Not included
No datasets are included. Public datasets must be obtained from their original sources while the synthetic datasets (SyntAC) are available separately on [Zenodo](https://doi.org/10.5281/zenodo.18491763).

## Before running
`main.py` expects a specific local folder layout for the training datasets. For the real+synthetic hybrid experiments (E3, E4), mixing is **not automated** and the real and synthetic subsets must be manually organized into the expected directory structure before training, so you will need to adapt the relevant paths/logic in `main.py` to your own data layout.
