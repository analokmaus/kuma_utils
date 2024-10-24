import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import scipy
from kuma_utils.preprocessing.utils import analyze_column
import matplotlib.pyplot as plt
import seaborn as sns


def _mean_std(arr):
    return arr.mean(), arr.std()


def make_demographic_table(
        df: pd.DataFrame,
        group_col: str,
        display_cols: list[str],
        categorical_cols: list[str] = [],
        numerical_cols: list[str] = [],
        categorical_cutoff: int = 5,
        handle_missing: str = 'value',
        categorical_omission_count: int = 50,
):
    '''
    Demographic table generator

    for numeric variables:
    - run KS test
        - if the varible follows normal distribution, run T test
        - if no, run Mann Whitney U test

    for categorical variables:
    - run chi-squared test
    '''
    res = []
    group_vals = df[group_col].unique()
    assert len(group_vals) == 2
    g1, g2 = group_vals
    g1_index = np.where(df[group_col] == g1)[0]
    g2_index = np.where(df[group_col] == g2)[0]
    res.append({
        '_item': 'N',
        '_type': 'numerical',
        '_ks_stat': None,
        '_stat_test': None,
        '_nan_info': False,
        g1: len(g1_index),
        g2: len(g2_index),
        'p-value': None
    })

    if len(categorical_cols) == 0 and len(numerical_cols) == 0:
        numerical_cols = []
        categorical_cols = []
        for col in display_cols:
            if col == group_col:
                continue
            if df[col].nunique() <= categorical_cutoff or analyze_column(df[col]) == 'categorical':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

    for col in categorical_cols:
        if handle_missing == 'value':
            col_arr = df[col].copy().fillna('NaN')
        elif handle_missing == 'ignore':
            col_arr = df[col].copy().dropna()
        else:
            col_arr = df[col].copy()
        val_cnt = col_arr.value_counts()
        if len(val_cnt) > categorical_omission_count:
            col_arr = col_arr.apply(
                lambda x: f'[{col}_other_categories]' if x in val_cnt.iloc[categorical_omission_count:].index else x)
            val_cnt = col_arr.value_counts()
        _le = {k: v for v, k in enumerate(val_cnt.index)}
        vals_all = col_arr.replace(_le).infer_objects(copy=False).values
        vals_g1 = vals_all[g1_index]
        vals_g2 = vals_all[g2_index]
        if len(vals_all) == 1:
            continue
        for dec_val, enc_val in _le.items():
            if dec_val == 0:
                continue
            pos_g1 = (vals_g1 == enc_val).sum()
            pos_g2 = (vals_g2 == enc_val).sum()
            pval = scipy.stats.chi2_contingency([
                [pos_g1, len(vals_g1) - pos_g1],
                [pos_g2, len(vals_g2) - pos_g2]
            ])[1]
            res.append({
                '_item': f'{col}={dec_val}, n(%)',
                '_type': 'categorical',
                '_ks_stat': None,
                '_stat_test': 'Chi2',
                '_nan_info': dec_val == 'NaN',
                g1: f'{pos_g1} ({pos_g1/len(vals_g1)*100:.1f}%)',
                g2: f'{pos_g2} ({pos_g2/len(vals_g2)*100:.1f}%)',
                'p-value': f'{pval:.3f}'
            })

    for col in numerical_cols:
        col_arr = df[col].dropna().copy()
        vals_all = df[col].values.reshape(-1)
        vals_g1 = vals_all[g1_index]
        vals_g2 = vals_all[g2_index]
        mean_all, std_all = _mean_std(vals_all)
        mean_g1, std_g1 = _mean_std(vals_g1)
        mean_g2, std_g2 = _mean_std(vals_g2)
        ks_res = scipy.stats.kstest(vals_all, 'norm', args=(mean_all, std_all))
        if ks_res.pvalue < 0.05:
            col_is_norm = False
        else:
            col_is_norm = True
        if col_is_norm:
            t_res = scipy.stats.ttest_ind(vals_g1, vals_g2, equal_var=False)
            res.append({
                '_item': f'{col}, mean(std)',
                '_type': 'numerical',
                '_ks_stat': ks_res.pvalue,
                '_stat_test': 'T',
                '_nan_info': False,
                g1: f'{mean_g1:.3f} ({std_g1:.3f})',
                g2: f'{mean_g2:.3f} ({std_g2:.3f})',
                'p-value': f'{t_res.pvalue:.3f}'
            })
        else:
            u_res = scipy.stats.mannwhitneyu(vals_g1, vals_g2, alternative='two-sided')
            res.append({
                '_item': f'{col}, mean(std)',
                '_type': 'numerical',
                '_ks_stat': ks_res.pvalue,
                '_stat_test': 'U',
                '_nan_info': False,
                g1: f'{mean_g1:.3f} ({std_g1:.3f})',
                g2: f'{mean_g2:.3f} ({std_g2:.3f})',
                'p-value': f'{u_res.pvalue:.3f}'
            })

    return pd.DataFrame(res)


def make_summary_table(
        df: pd.DataFrame,
        display_cols: list[str],
        categorical_cols: list[str] = [],
        numerical_cols: list[str] = [],
        categorical_cutoff: int = 5,
        handle_missing: str = 'value',
        categorical_omission_count: int = 50
):
    '''
    Summary table generator
    '''
    res = []
    res.append({
        '_item': 'N',
        '_type': 'numerical',
        '_nan_info': False,
        '_stat': len(df)
    })

    if len(categorical_cols) == 0 and len(numerical_cols) == 0:
        numerical_cols = []
        categorical_cols = []
        for col in display_cols:
            if df[col].nunique() <= categorical_cutoff or analyze_column(df[col]) == 'categorical':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

    for col in categorical_cols:
        if handle_missing == 'value':
            col_arr = df[col].copy().fillna('NaN')
        elif handle_missing == 'ignore':
            col_arr = df[col].copy().dropna()
        else:
            col_arr = df[col].copy()
        val_cnt = col_arr.value_counts()
        if len(val_cnt) > categorical_omission_count:
            col_arr = col_arr.apply(
                lambda x: f'[{col}_other_categories]' if x in val_cnt.iloc[categorical_omission_count:].index else x)
            val_cnt = col_arr.value_counts()
        _le = {k: v for v, k in enumerate(val_cnt.index)}
        vals_all = col_arr.replace(_le).infer_objects(copy=False).values
        if len(vals_all) == 1:
            continue
        for dec_val, enc_val in _le.items():
            if dec_val == 0:
                continue
            pos_all = (vals_all == enc_val).sum()
            res.append({
                '_item': f'{col}={dec_val}, n(%)',
                '_type': 'categorical',
                '_nan_info': dec_val == 'NaN',
                '_stat': f'{pos_all} ({pos_all/len(vals_all)*100:.1f}%)',
            })

    for col in numerical_cols:
        col_arr = df[col].dropna().copy()
        vals_all = df[col].values.reshape(-1)
        mean_all, std_all = _mean_std(vals_all)
        res.append({
            '_item': f'{col}, mean(std)',
            '_type': 'numerical',
            '_nan_info': False,
            '_stat': f'{mean_all:.3f} ({std_all:.3f})',
        })

    return pd.DataFrame(res)


def love_plot(
        data: pd.DataFrame,
        match_col: str,
        treatment_col: str,
        covariates: list[str],
        fig_params: dict = {},
        return_df: bool = False,
        title: str = "Covariate Balance",
        ):
    data_matched = data.query(f'{match_col} == 1')
    means_treated_before = data.loc[data[treatment_col] == 1, covariates].mean()
    means_control_before = data.loc[data[treatment_col] == 0, covariates].mean()
    means_treated_after = data_matched.loc[data_matched[treatment_col] == 1, covariates].mean()
    means_control_after = data_matched.loc[data_matched[treatment_col] == 0, covariates].mean()
    std_before = data[covariates].std()
    std_after = data_matched[covariates].std()
    mean_diffs = pd.concat([
        pd.DataFrame(
            (means_treated_before - means_control_before).abs()/std_before).rename(
                columns={0: 'value'}).reset_index().assign(name='before matching'),
        pd.DataFrame(
            (means_treated_after - means_control_after).abs()/std_after).rename(
                columns={0: 'value'}).reset_index().assign(name='after matching'),
    ])
    if return_df:
        return mean_diffs
    else:
        fig, ax = plt.subplots(**fig_params)
        sns.barplot(data=mean_diffs, x='value', y='index', hue='name', ax=ax)
        ax.axvline(0.1, color="red", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Absolute Standardized Mean Differences")
        ax.set_ylabel("Covariates")
        plt.tight_layout()
        return fig
