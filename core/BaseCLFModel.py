import joblib
import pandas as pd
from typing import List


class BaseCLFModel:
    
    def predict(self, df: pd.DataFrame, text_columns: List[str], target_column: str) -> pd.DataFrame:
        
        return NotImplementedError

    @staticmethod
    def return_text(df: pd.DataFrame, text_columns: List[str]) -> pd.Series:
        """
        joins text from text columns into one pd.Series
        
        :param df: original dataframe for inference
        :param text_columns: column with text relevant for inference
        
        :return pd.Series with all textual information in rows
        """
        text_all = []

        text_columns_amount = len(text_columns)

        if text_columns_amount == 0:
            raise ValueError("There is no text colums specified")

        elif text_columns_amount == 1:
            text_all = df[text_columns[0]].to_list()

        else:
            
            df_text = df[text_columns]
            df_text = df_text.applymap(
                lambda x: '' if not isinstance(x, str) else x
            )
            text_all = df_text.agg(' '.join, axis=1)

        return pd.Series(text_all)