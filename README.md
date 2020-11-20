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
┣ compat                        - Old version of kuma_utils for compatibility.
┣ visualization
    
┣ preprocessing
    ┣ xfeat                     - xfeat modifications.
        ┣ TargetEncoder
        ┣ Pipeline
    ┣ DistTransformer           - Distribution transformer for numerical features. 
┣ training
    ┣ Trainer                   - Amazing wrapper for scikit-learn API models.
    ┣ CrossValidator            - Amazing cross validation wrapper.
    ┣ LGBMLogger                - Logger callback for LightGBM/XGBoost/Optuna.
    ┣ StratifiedGroupKFold      - Stratified group k-fold split.
    ┣ optuna                    - optuna modifications.
        ┣ lighgbm               - Optune lightgbm integration with modifiable n_trials.
┣ metrics                       - Universal metrics
    ┣ SeWithFixedSp             - Sensitivity with fixed specificity.
    ┣ RMSE
    ┣ AUC
    ┣ Accuracy
    ┣ QWK
┣ torch
    ┣ model_zoo
        ┣ TabularNet            - Simple DNN for tabular data.
    ┣ TorchTrainer              - PyTorch Wrapper.
    ┣ EarlyStopping             - Early stopping for TorchTrainer (callback).
    ┣ TorchLogger               - Logger
    ┣ TemperatureScale          - Probability calibration for pytorch models.

```

# Tutorial
- [Train and validate scikit-learn API models](examples/Train_and_validate_models.ipynb)
- [Train PyTorch models](examples/Train_pytorch_models.ipynb)