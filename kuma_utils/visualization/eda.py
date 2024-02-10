import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from pprint import pprint, pformat
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns

from kuma_utils.preprocessing import analyze_column

try:
    import japanize_matplotlib
except ModuleNotFoundError:
    pass
try:
    import wandb
    WANDB = True
except ModuleNotFoundError:
    WANDB = False


def explore_data(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None,
        exclude_columns: List[str] = [],
        normalize: bool = True,
        histogram_n_bins: int = 20,
        use_wandb: bool = False,
        wandb_params: dict = {},
        plot: bool = True,
        scale_plot: Union[int, float] = 1.0,
        save_to: Union[str, Path] = None,
        verbose: bool = True,
    ) -> None:

    if use_wandb:
        if WANDB:
            wandb.init(**wandb_params)
            print('wandb export is enabled. Local plot will be disabled.')
            plot = False
        else:
            print('wandb is not installed.')
            use_wandb = False
  
    if verbose:
        print(f'data shape: {train_df.shape}, {test_df.shape if test_df is not None else None}\n')

    ''' Scan columns '''
    train_columns = train_df.columns.tolist()
    if test_df is not None:
        test_columns = test_df.columns.tolist()
        include_columns = set(train_columns) & set(test_columns)
        include_columns = list(include_columns - set(exclude_columns))
    else:
        include_columns = list(set(train_columns) - set(exclude_columns))
    n_columns = len(include_columns)
    if verbose:
        print(f'Included columns: \n{pformat(include_columns, compact=True)} ({n_columns})\n')
    if plot:
        fig_length = int(np.ceil(np.sqrt(n_columns)))
        fig = plt.figure(
            figsize=(fig_length*6*scale_plot, fig_length*3*scale_plot),
            tight_layout=True)
    
    ''' Check columns '''
    for icol, col in enumerate(include_columns):
        print(f'{icol:-{len(str(n_columns))+1}} {col}: ')
        vals = train_df[col]
        if test_df is not None:
            test_vals = test_df[col]

        column_type = analyze_column(vals)
        if column_type == 'numerical':
            nan_count = vals.isnull().sum()
            print(f'\n train NaN: {nan_count} ({nan_count/len(vals):.3f})')
            if test_df is not None:
                test_nan_count = test_vals.isnull().sum()
                print(f' test NaN: {test_nan_count} ({test_nan_count/len(test_vals):.3f}\n')
                bin_edges = np.histogram_bin_edges(
                    np.concatenate([vals.values, test_vals.values]), bins=histogram_n_bins)
            else:
                bin_edges = np.histogram_bin_edges(vals.values, bins=histogram_n_bins)
            
            if plot or use_wandb:
                if plot:
                    ax1 = fig.add_subplot(fig_length, fig_length, icol+1)
                elif use_wandb:
                    fig, ax1 = plt.subplots()

                sns.histplot(vals.dropna(), ax=ax1, label='train', bins=bin_edges,
                             element="step", stat="density", common_norm=False, kde=True)
                if test_df is not None:
                    sns.histplot(test_vals.dropna(), ax=ax1, label='test', bins=bin_edges,
                                 element="step", stat="density", common_norm=False, kde=True)
                plt.legend()
          
                if use_wandb:
                    wandb.log({f"{col}/distribution": wandb.Image(fig)})
                    plt.close()

        elif column_type == 'categorical':
            val_cnt = vals.value_counts(dropna=False)
            if test_df is not None:
                val_cnt = pd.concat(
                    [val_cnt, test_vals.value_counts(dropna=False)], 
                    axis=1).fillna(0)
                val_cnt.columns = ['train', 'test']
                if normalize:
                    val_cnt['train'] = val_cnt['train'] / val_cnt['train'].sum()
                    val_cnt['test'] = val_cnt['test'] / val_cnt['test'].sum()
                else:
                    val_cnt = val_cnt.astype(int)
            else:
                val_cnt = pd.DataFrame(val_cnt)
                val_cnt.columns = ['train']
                if normalize:
                    val_cnt['train'] = val_cnt['train'] / val_cnt['train'].sum()
                else:
                    val_cnt = val_cnt.astype(int)
            
            if use_wandb:
                wandb.log({f"{col}/count": wandb.Table(dataframe=val_cnt.reset_index())})

            print(f'\n n_classes={len(val_cnt)}')
            print(f'{val_cnt}\n')

            if plot or use_wandb:
                if plot:
                    ax1 = fig.add_subplot(fig_length, fig_length, icol+1)
                    ax1.set_title(col)
                elif use_wandb:
                    fig, ax1 = plt.subplots()

                train_uni = set(vals.unique().tolist())
                if test_df is not None:
                    test_uni = set(test_vals.unique().tolist())
                    common_uni = train_uni & test_uni
                    train_uni = train_uni - common_uni
                    test_uni = test_uni - common_uni
                    venn2(subsets=(
                        len(train_uni), len(test_uni), len(common_uni)),
                        set_labels=('train', 'test'), ax=ax1
                    )
                else:
                    venn2(subsets=(
                        0, 0, len(train_uni)),
                        set_labels=('train', 'train'), ax=ax1
                    )
                plt.legend()
                
                if use_wandb:
                    wandb.log({f"{col}/venn": wandb.Image(fig)})
                    plt.close()

    if plot:
        if save_to:
            fig.savefig(save_to)
        plt.show()
