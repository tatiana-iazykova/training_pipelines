from typing import Union, List, Any, Tuple

import pandas as pd
from nptyping import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from pipelines.build_tfidf_logreg import get_classification_report
from pipelines.registry import Registry
from utils.constants import RANDOM_STATE, NUM_MODELS


def find_best_model(
    X: Union[pd.Series, List[str]],
    y: List[Any]
) -> Tuple[NDArray, NDArray, pd.DataFrame, Pipeline]:

    registry = Registry()
    skf = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE, shuffle=True)

    pipe = registry.pipeline
    params = registry.params

    rscv = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_iter=NUM_MODELS,
        scoring='f1_weighted'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    rscv.fit(X=X_train, y=y_train)
    model = Pipeline([("vectoriser", TfidfVectorizer()), ("classifier", None)])
    model.set_params(**rscv.best_params_)
    model.fit(X_train, y_train)

    train_scores = cross_val_score(
        model, X=X_train, y=y_train, cv=skf, scoring='f1_weighted'
    )

    overall_scores = cross_val_score(
        model, X=X_test, y=y_test, cv=skf, scoring='f1_weighted'
    )

    y_pred = model.predict(X_test)

    df_classification_report = get_classification_report(y_test=y_test, y_pred=y_pred)

    return train_scores, overall_scores, df_classification_report, model

