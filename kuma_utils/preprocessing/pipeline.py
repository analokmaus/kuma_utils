import numpy as np
import pandas as pd


class PrepPipeline:
    '''
    Data Preprocessing Pipeline
    '''

    def __init__(self, transforms, target_col=None):
        self._transforms = transforms
        self._target_col = target_col
   
    def fit(self, input_df: pd.DataFrame) -> None:
        raise RuntimeError("fit() is not supported.")
    
    def fit_transform(self, input_df: pd.DataFrame) -> pd.DataFrame:  
        if self._target_col is not None:
            y = input_df[self._target_col].copy()
        else:
            y = None
            
        for transform in self._transforms:
            # Save df columns before transform
            if isinstance(input_df, pd.DataFrame):
                columns = input_df.columns
            else:
                columns = None
            if y is not None:
                try:
                    input_df = transform.fit_transform(input_df, y)
                except:
                    input_df = transform.fit_transform(input_df)
            else:
                input_df = transform.fit_transform(input_df)
            # Restore columns
            if (columns is not None) and isinstance(input_df, np.ndarray) and (len(columns) == input_df.shape[1]):
                input_df = pd.DataFrame(input_df, columns=columns)
        return input_df
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        for transform in self._transforms:
            if isinstance(input_df, pd.DataFrame):
                columns = input_df.columns
            else:
                columns = None
            input_df = transform.transform(input_df)
            if (columns is not None) and isinstance(input_df, np.ndarray) and (len(columns) == input_df.shape[1]):
                input_df = pd.DataFrame(input_df, columns=columns)
        return input_df
