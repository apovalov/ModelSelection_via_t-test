"""Solution for Kaggle AB1 Cross-Validation task."""
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm


def prepare_dataset(DATA_PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset.
    Load data, split into X and y, one-hot encode categorical

    Parameters
    ----------
    DATA_PATH: str :
        path to the dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        X and y
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["ID"], axis=1)
    y = df.pop("y").values

    # select only numeric columns
    X_num = df.select_dtypes(include="number")

    # select only categorical columns and one-hot encode them
    X_cat = df.select_dtypes(exclude="number")
    X_cat = pd.get_dummies(X_cat)

    # combine numeric and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.fillna(0).values

    return X, y


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Tuple[int, int]],
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 0,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Cross-validation score.

    Parameters
    ----------
    model: Callable :
        model to train (e.g. RandomForestRegressor)

    X: np.ndarray :

    y: np.ndarray :

    cv Union[int, Tuple[int, int]]:
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

    params_list: List[Dict] :
        list of model parameters

    scoring: Callable :
        scoring function (e.g. r2_score)

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    np.ndarray :
        cross-validation scores [n_models x n_folds]

    """
    if isinstance(cv, int):
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    elif isinstance(cv, tuple):
        cv_split = RepeatedKFold(
            n_splits=cv[0], n_repeats=cv[1], random_state=random_state
        )
    else:
        raise ValueError("cv should be either int or tuple")

    metrics = []
    for params in tqdm(params_list) if show_progress else params_list:
        metrics.append([])
        for train_index, test_index in cv_split.split(X):
            # Split
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train
            model.set_params(**params)
            model.fit(X_train, np.log1p(y_train))

            # Predict
            y_pred = np.expm1(model.predict(X_test))

            # Evaluate
            value = scoring(y_test, y_pred)
            metrics[-1].append(value)

    metrics = np.array(metrics)

    return metrics


def compare_models(
    cv: Union[int, Tuple[int, int]],
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
    alpha: float = 0.05,
    show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: Union[int, Tuple[int, int]] :
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    alpha: float :
        (Default value = 0.05)
        significance level for t-test

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            p_value,
            effect_sign
        }
    """
    metrics = cross_val_score(
        model=model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )

    result = []
    baseline_avg_score = metrics[0].mean()
    for i in range(1, len(params_list)):
        avg_score = metrics[i].mean()
        _, p_value = ttest_rel(metrics[0], metrics[i])

        if p_value < alpha and avg_score > baseline_avg_score:
            effect_sign = 1
        elif p_value < alpha and avg_score < baseline_avg_score:
            effect_sign = -1
        else:
            effect_sign = 0

        result.append(
            {
                "model_index": i,
                "avg_score": float(avg_score),
                "p_value": float(p_value),
                "effect_sign": effect_sign,
            }
        )

    result = sorted(result, key=lambda x: x["avg_score"], reverse=True)
    return result


def run() -> None:
    """Run."""

    data_path = "train.csv.zip"
    random_state = 42
    alpha = 0.05
    params_list = [
        {"max_depth": 10},  # baseline
        {"max_depth": 2},
        {"max_depth": 3},
        {"max_depth": 4},
        {"max_depth": 5},
        {"max_depth": 9},
        {"max_depth": 11},
        {"max_depth": 12},
        {"max_depth": 15},
    ]

    X, y = prepare_dataset(data_path)
    model = RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=random_state
    )

    # KFold
    cv = 5
    result = compare_models(
        cv=cv,
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        alpha=alpha,
        show_progress=True,
    )
    print("KFold")
    print(pd.DataFrame(result))

    # RepeatedKFold
    cv = (5, 3)
    result = compare_models(
        cv=cv,
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        alpha=alpha,
        show_progress=True,
    )
    print("RepeatedKFold")
    print(pd.DataFrame(result))


if __name__ == "__main__":
    run()
