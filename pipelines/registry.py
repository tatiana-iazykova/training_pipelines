from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class Registry:

    vectoriser_step = ('vectoriser', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True))

    logreg = LogisticRegression()
    multinomial_nb = MultinomialNB()
    random_forest = RandomForestClassifier()

    param_logreg = {
        'vectoriser__ngram_range': [(1, 2), (2, 2)],
        'classifier__C': (1.5, 1, 0.5),
        'classifier__penalty': ("l1", "l2"),
        'classifier__class_weight': ['balanced'],
        'classifier': [logreg]
    }

    param_multinomial_nb = {
        'vectoriser__ngram_range': [(1, 2), (2, 2)],
        'classifier__alpha': (0.01, 0.001),
        'classifier': [multinomial_nb]
    }

    param_random_forest = {
        'vectoriser__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'classifier__class_weight': ("balanced", "balanced_subsample"),
        'classifier': [random_forest]
    }

    pipeline = Pipeline([
        vectoriser_step,
        ('classifier', logreg)])

    params = [param_logreg, param_multinomial_nb, param_random_forest]
