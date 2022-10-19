import pandas as pd

def compute_percentage_of_suitable_data(df, target_column):
    percentage_of_observations = df[target_column].isna().sum() / len(df)
    return (1 - percentage_of_observations) * 100


def check_value_counts(df, target_column, threshold=2):
    value_cnt_df = pd.DataFrame(df[target_column].value_counts())
    return len(value_cnt_df[value_cnt_df.values > threshold]) / len(value_cnt_df) * 100