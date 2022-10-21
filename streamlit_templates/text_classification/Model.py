import joblib
import pandas as pd
from typing import List


class Model:

    def __init__(self):
        self.model = joblib.load('model.joblib')
    
    def predict(self, df: pd.DataFrame, text_columns: str, target_column: str) -> pd.DataFrame:

        data = self.return_text(df=df, text_columns=text_columns)

        df[target_column] = self.model.predict(data)

        return df

    @staticmethod
    def return_text(df: pd.DataFrame, text_columns: List[str]) -> pd.Series:
        text_all = []

        if len(text_columns) == 1:
            text_all = df[text_columns[0]].to_list()

        elif len(text_columns) > 1:

            for _, row in text_columns.iterrows():
                text = []

                for col in text_columns:
                    if type(row[col]) == str:
                        text.append(row[col])
                text_all.append(' '.join(text).strip())
        else:
            raise ValueError("There is no text colums specified")

        return pd.Series(text_all)