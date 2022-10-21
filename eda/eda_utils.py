from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt


def compute_percentage_of_suitable_data(df: pd.DataFrame, target_column: str, full_length: int) -> float:
    """
    counts the percentage of observations for which target is present
    :param df: pd.DataFrame which is going to be used for modeling
    :param target_column: column name where to find targets
    :param full_length: dataframe initial length to compute correct statistics
    :return percentage of data eligible for modeling

    Example:

    >>> df
            text        target
        0    aaaaaa       NaN
        1      eeee       1
        2   ggggggg       1
        3     hhhhh       0
        4      ffff       1
        5     dddd        0
        6    ccccc        NaN  

    >>> target_column = "target"
    >>> full_length = 20
    >>> compute_percentage_of_suitable_data(df=df, target_column=target_column, full_length=full_length)
    20.00
    """

    percentage_of_observations = df[target_column].isna().sum() / full_length
    return (1 - percentage_of_observations) * 100


def check_value_counts(df: pd.DataFrame, target_column: str, threshold: int = 2) -> Tuple[pd.DataFrame, float]:
    """
    counts the percentage of observations that are well-represented in data
    :param df: pd.DataFrame which is going to be used for modeling
    :param target_column: column name where to find targets
    :param threshold: minimal observation count
    :return dataframe with observations eligible for modelling, percentage of labels eligible for modeling

    Example:

    >>> df
        text       target
    0    aaaaaa       1
    1      eeee       0
    2   ggggggg       0
    3     hhhhh       0
    4      ffff       1
    5     dddd        1
    6    ccccc        1

    >>> target_column = "target"
    >>> threshold = 2
    >>> df, value_counts_check = check_value_counts(df=df, target_column=target_column, threshold=threshold)
    >>> df
        text       target
    0    aaaaaa       1
    1      eeee       0
    2   ggggggg       0
    3     hhhhh       0
    4      ffff       1
    5     dddd        1
    6    ccccc        1
    >>> value_counts_check
        100.00
    """

    value_cnt_df = pd.DataFrame(df[target_column].value_counts())
    df = df[df[target_column].isin(list(value_cnt_df[value_cnt_df.values > threshold].index))]
    return df, len(value_cnt_df[value_cnt_df.values > threshold]) / len(value_cnt_df) * 100


def render_pie_chart(df: pd.DataFrame, column_name: str) -> plt.Axes:
    """
    creates pie chart based on column proportion 
    :param df: pd.DataFrame which is going to be used for modeling
    :param target_column: column name where to find targets
    :return matplotlib pie chart object for further rendering

    """
    value_counts_data = df[column_name].value_counts()
    _ = plt.pie(
        value_counts_data, 
        labels=value_counts_data.keys(), 
        autopct="%1.1f%%", 
        pctdistance=0.5, 
        )
    _ = plt.title("Label distribution")

    return plt