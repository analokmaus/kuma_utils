import numpy as np
import pandas as pd
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from kuma_utils.training import Trainer
from kuma_utils.preprocessing import PreprocessingTemplate, PrepPipeline, SelectCategorical, SelectNumerical


class PropensityScoreMatching:
    '''
    Propensity score matching with various backend models
    '''
    def __init__(self,
                 match_cols: list[str],
                 group_col: str,
                 return_zscore: bool = True,
                 categorical_encoder: PreprocessingTemplate = PrepPipeline([SelectCategorical()]),
                 numerical_encoder: PreprocessingTemplate = PrepPipeline([SelectNumerical()]),
                 matching_method: str = 'hungarian',
                 model: Any = LogisticRegression,
                 trainer_params: dict[dict] = {'params': {}, 'fit_params': {}},
                 fit_method: str = 'fit',
                 caliper: str | float = 'auto'):
        self.match_cols = match_cols
        self.group_col = group_col
        self.return_zscore = return_zscore
        self.matching_method = matching_method
        self._le = LabelEncoder()
        self._cat_enc = categorical_encoder
        self._num_enc = numerical_encoder
        self.trainer = Trainer(model)
        self.trainer_params = trainer_params
        self.fit_method = fit_method
        self.caliper = caliper
        assert self.matching_method in ['greedy', 'hungarian']
        assert self.caliper == 'auto' or isinstance(caliper, float)
        assert self.fit_method in ['fit', 'cv']

    def _match(self, ps1, ps2, caliper):
        distance_matrix = np.abs(ps1 - ps2)
        if self.matching_method == 'greedy':
            ps1_index = []
            ps2_index = []
            while np.min(distance_matrix) < caliper:
                # get index of minimum distance element
                i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
                ps1_index.append(i)
                ps2_index.append(j)
                distance_matrix[i, :] = 1
                distance_matrix[:, j] = 1
        elif self.matching_method == 'hungarian':
            row_idx, col_idx = linear_sum_assignment(distance_matrix)
            valid_idx = distance_matrix[row_idx, col_idx] < caliper
            ps1_index, ps2_index = row_idx[valid_idx], col_idx[valid_idx]
        del distance_matrix
        return ps1_index, ps2_index

    def run(self, df: pd.DataFrame):
        assert df[self.group_col].nunique() == 2
        X_match = df[self.match_cols].copy()
        X_match = pd.concat([
            self._cat_enc.fit_transform(X_match).reset_index(drop=True),
            self._num_enc.fit_transform(X_match).reset_index(drop=True)], axis=1)
        y_match = self._le.fit_transform(df[self.group_col].copy())

        if self.fit_method == 'fit':
            self.trainer.fit(
                train_data=(X_match, y_match),
                valid_data=(X_match, y_match),
                **self.trainer_params)
            X_match['__z_score'] = self.trainer.predict_proba(X_match)[:, 1]
        elif self.fit_method == 'cv':
            self.trainer.cv(
                data=(X_match, y_match),
                **self.trainer_params)
            X_match['__z_score'] = np.mean(self.trainer.smart_predict(X_match), axis=0)

        if self.caliper == 'auto':
            caliper = X_match['__z_score'].std() * 0.2
        else:
            caliper = self.caliper
        zero_index, one_index = self._match(
            X_match.loc[y_match == 0, '__z_score'].values.reshape(-1, 1),
            X_match.loc[y_match == 1, '__z_score'].values,
            caliper=caliper)
        if self.return_zscore:
            df['_z_score'] = X_match['__z_score'].copy()
        matched_data = pd.concat([
            df.loc[y_match == 0].iloc[zero_index],
            df.loc[y_match == 1].iloc[one_index],
        ], axis=0).reset_index(drop=True)

        return matched_data

    def plot_feature_importance(self, df: pd.DataFrame):
        assert df[self.group_col].nunique() == 2
        X_match = df[self.match_cols].copy()
        X_match = pd.concat([
            self._cat_enc.fit_transform(X_match).reset_index(drop=True),
            self._num_enc.fit_transform(X_match).reset_index(drop=True)], axis=1)
        y_match = self._le.fit_transform(df[self.group_col].copy())
        self.trainer.plot_feature_importance(
            importance_type='permutation', fit_params={'X': X_match, 'y': y_match})
