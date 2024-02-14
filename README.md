# Kuma's Toolkit 2024

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
- Convenient functions for basic biostatistical analysis.

## What's new
- Wandb integration
- Upgrade to newer backend libraries
- Integration of TensorboardLogger into TorchLogger
- Automated hyperparameter tuning for lightgbm/xgboost/catboost.cv()

## Work in progress
- Multi-node DDP

# Setup
## With pip
```bash
pip install git+https://github.com/analokmaus/kuma_utils.git@v0.4.0  # Stable
pip install git+https://github.com/analokmaus/kuma_utils.git@master  # Latest
```
**IMPORTANT**
For Apple silicon users, there will be an error building lightgbm. 
Please install lightgbm with the following command and then install kuma_utils.
```bash
pip install --no-binary lightgbm --config-settings=cmake.define.USE_OPENMP=OFF  'lightgbm==4.3.0'
pip install git+https://github.com/analokmaus/kuma_utils.git
```

## With poetry
```bash
git clone https://github.com/analokmaus/kuma_utils.git
cd kuma_utils
poetry install
```
or simply,
```bash
poetry add git+https://github.com/analokmaus/kuma_utils.git
```

## Alternative installation methods
WIP

# Tutorials
- [Exploratory data analysis](examples/Exploratory_data_analysis.ipynb)
- [Data preprocessing](examples/Data_preprocessing.ipynb)
- [Train and validate scikit-learn API models](examples/Train_and_validate_models.ipynb)
- [Train pytorch models on single GPU](examples/Train_CNN_model.ipynb)
- [Train pytorch models on multiple GPU](examples/Train_CNN_distributed.py)
- [Statistical analysis (propensity score matching)](examples/Statistical_analysis.ipynb)

# Directory
```
┣ visualization
┃   ┣ explore_data              - Simple exploratory data analysis.
┃
┣ preprocessing
┃   ┣ SelectNumerical            
┃   ┣ SelectCategorical 
┃   ┣ DummyVariable 
┃   ┣ DistTransformer           - Distribution transformer for numerical features. 
┃   ┣ LGBMImputer               - Regression imputer for missing values using LightGBM.
┃
┣ stats
┃   ┣ make_demographic_table    - Automated demographic table generator.
┃   ┣ PropensityScoreMatching   - Fast and capable of using all sklearn API models as a backend.
┃
┣ training
┃   ┣ Trainer                   - Wrapper for scikit-learn API models.
┃   ┣ CrossValidator            - Ccross validation wrapper.
┃   ┣ LGBMLogger                - Logger callback for LightGBM/XGBoost/Optuna.
┃   ┣ StratifiedGroupKFold      - Stratified group k-fold split.
┃   ┣ optuna                    - optuna modifications.
┃
┣ metrics                       - Universal metrics
┃   ┣ SensitivityAtFixedSpecificity
┃   ┣ RMSE
┃   ┣ Pearson correlation coefficient
┃   ┣ R2 score
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