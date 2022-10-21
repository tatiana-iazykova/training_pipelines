import cleanlab
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Any
from models.build_tfidf_logreg import build_pipeline


def clean_duplicates(df: pd.DataFrame) -> Tuple[int, float, pd.DataFrame]:
    """
    cleans dataset from dupluicated rows
    :param df: pd.DataFrame which is going to be used for modeling

    :return df's original length, number of rows removed and cleaned dataframe

    Example

    >>> df
            text        target  age
        0    aaaaaa       1     12
        1      eeee       1     124
        2   ggggggg       1     0
        3     hhhhh       0     8    
        4      ffff       1     10
        5     aaaaaa      1     12
        6    ccccc        NaN   NaN   
    >>> full_length, cnt_duplicates, df = clean_duplicates(df=df)
    >>> df
                text        target  age
        0    aaaaaa       1     12
        1      eeee       1     124
        2   ggggggg       1     0
        3     hhhhh       0     8    
        4      ffff       1     10
        5    ccccc        NaN   NaN 
    >>> full_length
    7
    >>> cnt_duplicates
    1.0
    """
    full_length = len(df)
    cnt_duplicates = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    return full_length, cnt_duplicates, df


def clean_relevant_duplicates(df: pd.DataFrame, text_columns: List[str], target_column: str) -> Tuple[int, float, pd.DataFrame]:
    """
    removes irrelevant columns and once again cleans the duplicates

    :param df: pd.DataFrame which is going to be used for modeling
    :param text_columns: list of column names that contain text relevant to the task
    :param target_column: str, column where markup is stored

    :return dataframe length before removing duplicates, number of rows removed and cleaned dataframe

    Example

    >>> df
            text        target  age
        0    aaaaaa       1     12
        1      eeee       1     124
        2   ggggggg       1     0
        3     hhhhh       0     8    
        4      ffff       1     10
        5     hhhhh       1     NaN   
    >>> text_columns
    ['text']
    >>> target_column
    'target'
    >>> relevant_length, cnt_relevant_duplicates, df = clean_relevant_duplicates(df=df, text_columns=text_columns, target_column=target_column)
    >>> df
            text        target  
        0    aaaaaa       1     
        1      eeee       1     
        2   ggggggg       1     
        3     hhhhh       0         
        4      ffff       1             
    >>> relevant_length
    6
    >>> cnt_duplicates
    1.0
    """
    relevant_length = len(df)
    cnt_relevant_duplicates = df[text_columns +
                                 [target_column]].duplicated().sum()
    df = df[text_columns + [target_column]
            ].drop_duplicates().reset_index(drop=True)
    return relevant_length, cnt_relevant_duplicates, df


def return_text_and_targets(df: pd.DataFrame, text_columns: List[str], target_column: str) -> Tuple[pd.Series, List[Any]]:
    """
    separates texts and targets, if there are a few text columns, joins text in them 

    :param df: pd.DataFrame which is going to be used for modeling
    :param text_columns: list of column names that contain text relevant to the task
    :param target_column: str, column where markup is stored

    :return pd.Series with text and list of targets

    Example

    >>> df
            text        target  
        0    aaaaaa       1     
        1      eeee       1     
        2   ggggggg       1     
        3     hhhhh       0         
        4      ffff       1  

    >>> text_columns
    ['text']
    >>> target_column
    'target'
    >>> X, y = return_text_and_targets(df=df, text_columns=text_columns, target_column=target_column)
    >>> X          
    0    aaaaaa          
    1      eeee            
    2   ggggggg            
    3     hhhhh                
    4      ffff         
    >>> y
    [1, 1, 1, 0, 1]
    """
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

    return pd.Series(text_all), df[target_column].to_list()


def analyse_data_annotation(X: pd.Series, y: List[Any], threshold: int) -> Tuple[float, pd.DataFrame]:
    """
    analyses data markup

    :param X: pd.Series which contains texts
    :param y: list of targets
    :param threshold: minimal number of observations to consider

    :return data quality score and full report

    Example

    >>> X
    0   hi there...          
    1   omg u wont't believe            
    2   hello there            
    3   as requested                
    4   ffff
    
    1 x 1500 
    >>> y
    ['ham', 'spam', 'spam', 'ham', ...., 'spam'] 
    >>> len(y)
    1500
    >>> threshold
    5
    >>> data_quality_score, report = analyse_data_annotation(X=X, y=y, threshold=threshold)
    >>> data_quality_score
    0.9986
    >>> report
        Class Name  Class Index  Label Issues  Inverse Label Issues  Label Noise  Inverse Label Noise  Label Quality Score
    0       spam            1             7                     9     0.010920             0.013997             0.989080
    1        ham            0             9                     7     0.001993             0.001551             0.998007
    """
    le = LabelEncoder()

    labels = le.fit_transform(y=y)
    pipe = build_pipeline()

    cl = cleanlab.classification.CleanLearning(
        pipe, find_label_issues_kwargs={
            'min_examples_per_class': threshold,
        }
    )

    cl.fit(X=X, labels=labels)

    hs = cleanlab.dataset.health_summary(
        labels, confident_joint=cl.confident_joint, class_names=le.classes_, verbose=False)

    data_quality_score = hs['overall_label_health_score']
    report = hs['classes_by_label_quality']

    return data_quality_score, report
