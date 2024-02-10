# Kuma-san's Toolkit 2024

```
　 　 　┼╂┼
　 　 ∩＿┃＿∩
    |ノ      ヽ
   /   ●    ● |
  |     (_●_) ミ        < There is absolutely no warranty. >
 彡､     |∪|  ､｀＼ 
/ ＿＿   ヽノ /´>  )
(＿＿＿）    / (_／
```

# Overview
Using this library, you can:

- Simplify the structuring of table data and feature engineering
- implify the training and hyperparameter search for ML tools with Sklearn API (including sklearn, lightgbm, catboost, etc.)
- Simplify the training of Pytorch models (including the use of amp and parallelization across multiple GPUs)
- Customize training with our own Hook/Callback interface (such as Earlystop, logging functions integrated with wandb, etc.)
- Automated exploratory data analysis

## What's new
- Wandb integration
- Upgrade to newer backend libraries
- Integration of TensorboardLogger into TorchLogger
- Automated hyperparameter tuning for lightgbm/xgboost.cv()

## Work in progress
- Multi-node DDP

# Setup
## With Poetry
```
git clone https://github.com/analokmaus/kuma_utils.git
cd kuma_utils
poetry install
```

## Alternative installation methods
WIP

# Tutorials
- [Exploratory data analysis](examples/Exploratory_data_analysis.ipynb)
- [Data preprocessing](examples/Data_preprocessing.ipynb)
- [Train and validate scikit-learn API models](examples/Train_and_validate_models.ipynb)
- [Train pytorch models on single GPU](examples/Train_CNN_model.ipynb)
- [Train pytorch models on multiple GPU](examples/Train_CNN_distributed.py)

# Directory
```
┣ visualization
┃   ┣ explore_data              - Simple exploratory data analysis.
┃
┣ preprocessing
┃   ┣ DistTransformer           - Distribution transformer for numerical features. 
┃   ┣ LGBMImputer               - Regression imputer for missing values using LightGBM.
┃
┣ training
┃   ┣ Trainer                   - Amazing wrapper for scikit-learn API models.
┃   ┣ CrossValidator            - Amazing cross validation wrapper.
┃   ┣ LGBMLogger                - Logger callback for LightGBM/XGBoost/Optuna.
┃   ┣ StratifiedGroupKFold      - Stratified group k-fold split.
┃   ┣ optuna                    - optuna modifications.
┃       ┣ lightgbm               - Optune lightgbm integration with modifiable n_trials.
┃
┣ metrics                       - Universal metrics
┃   ┣ SensitivityAtFixedSpecificity
┃   ┣ RMSE
┃   ┣ AUC
┃   ┣ Accuracy
┃   ┣ QuandricWeightKappa
┃
┣ torch
    ┣ lr_scheduler
    ┃   ┣ ManualScheduler
    ┃   ┣ CyclicCosAnnealingLR
    ┃   ┣ CyclicLinearLR
    ┃   
    ┣ optimizer
    ┃   ┣ SAM
    ┃ 
    ┣ modules
    ┃   ┣ Mish
    ┃   ┣ AdaptiveConcatPool2d/3d
    ┃   ┣ GeM
    ┃   ┣ CBAM2d
    ┃   ┣ GroupNorm1d/2d/3d
    ┃   ┣ convert_groupnorm     - Convert all BatchNorm to GroupNorm.
    ┃   ┣ etc...
    ┃ 
    ┣ TorchTrainer              - PyTorch Trainer.
    ┣ EarlyStopping             - Early stopping callback for TorchTrainer. Save snapshot when best score is achieved.
    ┣ SaveEveryEpoch            - Save snapshot at the end of every epoch.
    ┣ SaveSnapshot              - Snapshot callback.
    ┣ SaveAverageSnapshot       - Moving average snapshot callback.
    ┣ TorchLogger               - Logger
    ┣ TensorBoardLogger         - TensorBoard Logger
    ┣ SimpleHook                - Simple train hook for almost any tasks (see tutorial).
    ┣ TemperatureScaler         - Probability calibration for pytorch models.

```

# License
The source code in this repository is released under the MIT license.