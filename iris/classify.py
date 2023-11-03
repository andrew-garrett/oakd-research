#################### IMPORTS ####################
#################################################


from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import xgboost as xgb
from scipy.stats import randint, uniform  # type: ignore
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# from iris.data import IrisDataset

#################### TABULAR DATA CLASSIFIER ####################
#################################################################


def report_best_scores(results: Dict[str, np.ndarray], n_top: int = 3) -> None:
    """
    Print the summary statistics for the top n models from a hyperparameter search

    Arguments:
        - results: the cv_results_ attribute of GridSearchCV or RandomizedSearchCV object (sklearn.model_selection)
        - n_top: the number of models to display information from
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.5f} (std: {1:.5f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


def train_xgb(
    X: Any,
    y: Any,
    objective: str = "binary:logistic",
    eval_set: Optional[Sequence[Tuple[Any, Any]]] = None,
) -> None:
    """
    Train an XGBoost model on a dataset formatted for sklearn models, seeking
    to minimize a given objective

    Arguments:
        - X: the features of the dataset
        - y: the targets of the dataset
        - objective: a string indicating the task:loss_fn ("multi:softprob" for multi-class classification)
        - eval_set: a test set, if desired
    """
    xgb_model = xgb.XGBClassifier(
        objective=objective, early_stopping_rounds=10, random_state=42
    )
    xgb_model.fit(X, y, eval_set=eval_set, verbose=0)

    y_pred = xgb_model.predict(X)

    print(confusion_matrix(y, y_pred))


def tune_xgb(
    X: Any,
    y: Any,
    objective: str = "binary:logistic",
    eval_set: Optional[Sequence[Tuple[Any, Any]]] = None,
) -> None:
    """
    Tune an XGBoost model on a dataset formatted for sklearn models, seeking
    to minimize a given objective

    Uses sklearn.model_selection.RandomizedSearchV

    Arguments:
        - X: the features of the dataset
        - y: the targets of the dataset
        - objective: a string indicating the task:loss_fn
        - eval_set: a test set, if desired
    """
    xgb_model = xgb.XGBClassifier(
        objective=objective, early_stopping_rounds=10, random_state=42
    )

    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4),
    }

    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=params,
        random_state=42,
        n_iter=200,
        cv=3,
        n_jobs=1,
        return_train_score=True,
    )

    search.fit(X, y, eval_set=eval_set, verbose=1)

    report_best_scores(search.cv_results_, 1)

    if eval_set is not None:
        y_pred = search.predict(eval_set[0][0])
        print(confusion_matrix(eval_set[0][1], y_pred))


# if __name__ == "__main__":
# print()
# print("Train an xgb model (binary classification)")
# cancer = load_breast_cancer()
# X = cancer.data
# y = cancer.target
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# train_xgb(X_train, y_train, objective="binary:logistic", eval_set=[(X_val, y_val)])

# print()
# print("Train an xgb model (multi-class classification)")
# wine = load_wine()
# X = wine.data
# y = wine.target
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# train_xgb(X_train, y_train, objective="multi:softprob", eval_set=[(X_val, y_val)])

# print()
# print("Tune an xgb model (classification)")

# # Turn down for faster run time
# n_samples = 5000
# X, y = fetch_20newsgroups_vectorized(subset="all", return_X_y=True)
# X = X[:n_samples]
# y = y[:n_samples]
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# tune_xgb(X_train, y_train, objective="multi:softprob", eval_set=[(X_val, y_val)])

# iris_dataset = IrisDataset()
# X, y = iris_dataset.X, iris_dataset.y
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# tune_xgb(X_train, y_train, objective="binary:logistic", eval_set=[(X_val, y_val)])
