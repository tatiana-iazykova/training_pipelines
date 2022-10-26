import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from typing import Union, Any, List, Tuple
from nptyping import NDArray
from utils.constants import RANDOM_STATE


def build_pipeline() -> Pipeline:
    """
    function that builds sklearn pipeline with tfidf as vectoriser and logistic regression as model

    :return sklearn pipeline
    """

    steps = [
        ('vectoriser', TfidfVectorizer()),
        ('logreg', LogisticRegression(class_weight='balanced', n_jobs=-1))
    ]

    pipe = Pipeline(steps)

    return pipe


def fit_model(X: Union[pd.Series, List[str]], y: List[Any]) -> Tuple[Pipeline, NDArray]:
    """
    Computes f1-weighted cross validation on 5 stratified folds and fits model

    :param X: column or list of texts
    :param y: list of targets

    :return scores on cross validation and model fitted on the whole data
    """
    stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline()

    train_scores = cross_val_score(
        pipe, X=X, y=y, cv=stf, scoring='f1_weighted')

    pipe.fit(X=X, y=y)

    return pipe, train_scores


def get_classification_report(y_test: Union[List[Any], NDArray], y_pred: Union[List[Any], NDArray]) -> pd.DataFrame:
    """
    calcuates classification report and transforms it to a pretty dataframe

    :param y_test: true targets
    :param y_pred: model predictions

    :return: pretty dataframe
    """
    report = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report['support'] = df_classification_report['support'].astype(int)
    df_classification_report['support'].iloc[2] = ''
    df_classification_report['support'] = df_classification_report['support'].astype(str)
    df_classification_report['precision'] = df_classification_report['precision'].round(2)
    df_classification_report['recall'] = df_classification_report['recall'].round(2)
    df_classification_report['f1-score'] = df_classification_report['f1-score'].round(2)

    return df_classification_report


def build_tfidf_logreg(
    X: Union[pd.Series, List[str]], 
    y: List[Any]
    ) -> Tuple[NDArray, NDArray, pd.DataFrame, Pipeline]:
    """
    splits data into train and test, fits model and returns all metrics

    :param X: list or series of textual features
    :param y: list of targets

    :return train metrics, full data metrics, classification report on test data, fitted model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    model, train_scores = fit_model(X=X_train, y=y_train)
    _, overall_scores = fit_model(X=X, y=y)

    y_pred = model.predict(X_test)

    df_classification_report = get_classification_report(y_test=y_test, y_pred=y_pred)

    return train_scores, overall_scores, df_classification_report, model
