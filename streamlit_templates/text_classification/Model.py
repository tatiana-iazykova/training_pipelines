import joblib
import pandas as pd
from typing import List
from core.BaseCLFModel import BaseCLFModel


class Model(BaseCLFModel):

    def __init__(self):
        self.model = joblib.load('model.joblib')
    
    def predict(self, df: pd.DataFrame, text_columns: List[str], target_column: str) -> pd.DataFrame:
        """
        joins all specified text fields and runs inference

        :param df: dataframe on what to run inference
        :param text_columns: list of column names where relevant text data is stored
        :param target_column: target name

        :return dataframe with predictions
        """

        data = self.return_text(df=df, text_columns=text_columns)

        df[target_column] = self.model.predict(data)

        return df