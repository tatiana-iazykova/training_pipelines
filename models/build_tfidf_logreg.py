import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline


def build_pipeline():

    steps = [
        ('vectoriser', TfidfVectorizer()),
        ('logreg', LogisticRegression(class_weight='balanced', n_jobs=-1))
    ]

    pipe = Pipeline(steps)

    return pipe


def fit_model(X, y):
    stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = build_pipeline()

    train_scores = cross_val_score(
        pipe, X=X, y=y, cv=stf, scoring='f1_weighted')

    pipe.fit(X=X, y=y)

    return pipe, train_scores


def get_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report['support'] = df_classification_report['support'].astype(int)
    df_classification_report['support'].iloc[2] = ''
    df_classification_report['support'] = df_classification_report['support'].astype(str)
    df_classification_report['precision'] = df_classification_report['precision'].round(2)
    df_classification_report['recall'] = df_classification_report['recall'].round(2)
    df_classification_report['f1-score'] = df_classification_report['f1-score'].round(2)

    return df_classification_report


def build_tfidf_logreg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model, train_scores = fit_model(X=X_train, y=y_train)
    _, overall_scores = fit_model(X=X, y=y)

    y_pred = model.predict(X_test)

    df_classification_report = get_classification_report(y_test=y_test, y_pred=y_pred)

    return train_scores, overall_scores, df_classification_report, model
