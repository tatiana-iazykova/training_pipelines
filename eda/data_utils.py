import cleanlab
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd


def return_text_and_targets(df, text_columns, target_column):
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


def analyse_data_annotation(X, y, threshold):
    le = LabelEncoder()

    labels =le.fit_transform(y=y)

    steps = [
                ('vectoriser', TfidfVectorizer()),
                ('logreg', LogisticRegression(class_weight='balanced', n_jobs=-1))
            ]

    pipe = Pipeline(steps)

    cl = cleanlab.classification.CleanLearning(
        pipe, find_label_issues_kwargs={
            'min_examples_per_class': threshold,
            # "cv_n_folds": 
            }
        )

    cl.fit(X=X, labels=labels)

    hs = cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint, class_names=le.classes_, verbose=False)

    data_quality_score = hs['overall_label_health_score']
    report = hs['classes_by_label_quality']

    return data_quality_score, report