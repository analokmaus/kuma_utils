# Kuma-san's Toolkit

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

# Environment
`pip install -r reqirements.txt`
## Optional requirements
### xfeat
`pip install -q https://github.com/pfnet-research/xfeat/archive/master.zip`
### Category Encoders
`pip install category_encoders`
### CuPy
`WIP`
### NVIDIA apex
`WIP`


# Directory
```
┣ common
┣ compat                        - Old version of kuma_utils for compatibility.
┣ visualization
    ┣ ks_test                   - Kolmogorov-Smirnov test.
    ┣ explore_dataframe         - Automated EDA (WIP).
    ┣ plot_calibration_curve    - Plot calibation curve.
┣ preprocessing
    ┣ xfeat                     - xfeat modifications.
        ┣ TargetEncoder
        ┣ Pipeline
    ┣ DistTransformer           - Distribution transformer for numerical features. 
┣ training
    ┣ Trainer                   - Amazing wrapper for scikit-learn API models.
    ┣ LGBMLogger                - Logger callback for LightGBM/XGBoost.
    ┣ StratifiedGroupKFold      - Stratified group k-fold split.
    ┣ optuna                    - optuna modifications.
        ┣ lighgbm               - Optune lightgbm integration with modifiable n_trials.
┣ metrics                       - Universal metric class for whatever library.
    ┣ SeUnderSp                 - Sensitivity with specificity fixed.
    ┣ RMSE
    ┣ AUC
    ┣ Accuracy
    ┣ QWK
┣ torch                       
    ┣ datasets
        ┣ category2embedding    - Calculate optimal embedding dimensions of categorical features.
        ┣ Numpy2Dataset         - Convert numpy.array to torch.tensor.
    ┣ logger
        ┣ Logger                - Export TensorBoard logs.
    ┣ models
        ┣ TabularNet            - Simple DNN for tabular data.
    ┣ snapshot                  - Snapshot I/O.
    ┣ training
        ┣ TorchTrainer          - PyTorch Wrapper. See examples below.
        ┣ TrochCV               - Simple cross validation wrapper for TorchTrainer.
        ┣ DummyStopper          - Dummy stopper for TorchTrainer.
        ┣ EarlyStopping         - Early stopping for TorchTrainer.
        ┣ DummyEvent            - Dummy event for TorchTrainer.
    ┣ temperature_scaling.py    - Probability calibration for pytorch models.

```