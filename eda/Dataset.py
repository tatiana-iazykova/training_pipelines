import os
from typing import BinaryIO

import pandas as pd


class PandasDataset():
    """
    Dataset class that allows you to convert multiple formats to 1
    """

    def __init__(self, df_object: BinaryIO):

        self.valid_data_types = {
            '.csv': self._read_csv,
            '.tsv': self._read_csv,
            '.xls': self._read_excel,
            '.xlsx': self._read_excel,
        }

        self.data = self.read_data(fileobject=df_object)

    def read_data(self, fileobject: BinaryIO) -> pd.DataFrame:
        """
        Given the path to the file returns extension of that file

        Example:
        path: "../input/some_data.csv"
        :return: ".csv"
        """
        path = fileobject.name
        _, extension = os.path.splitext(path)
        if extension.lower() in self.valid_data_types:
            return self.valid_data_types[extension](fileobject=fileobject, extension=extension)
        else:
            raise ValueError(f"Your data type ({extension}) is not supported, please convert your dataset "
                             f"to one of the following formats {list(self.valid_data_types.keys())}.")

    @staticmethod
    def _read_csv(fileobject: BinaryIO, extension: str) -> pd.DataFrame:
        """
        Reads a csv file given its path
        :param path: "../../some_file.csv"
        :return: dataframe
        """
        sep = ','
        if extension == '.tsv':
            sep = '\t'

        return pd.read_csv(filepath_or_buffer=fileobject, sep=sep, encoding="utf-8")

    @staticmethod
    def _read_excel(fileobject: BinaryIO, extension: str) -> pd.DataFrame:
        """
        Reads a xls or xlsx file given its path
        :param path: "../../some_file.xlsx"
        :return: dataframe
        """
        engine = 'openpyxl'
        if extension == '.xls':
            engine = None

        return pd.read_excel(io=fileobject, engine=engine)
