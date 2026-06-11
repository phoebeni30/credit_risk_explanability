from typing import Tuple,  Dict, Union, Any, Callable
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

import optuna
from optuna.samplers import TPESampler

import pandas as pd
import numpy as np

from .training import create_pipeline, ks_threshold
from .config import hyperparam_spaces


def objective(
    trial: optuna.trial.Trial,
    model_class: BaseEstimator,
    pipeline_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    score_func: Union[str, Callable],
    param_space: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: int,
    seed_number: int,
):
    params = {}
    for name, values in param_space.items():
        if values["type"] == "int":
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_int(name, **values_cp)
        elif values["type"] == "categorical":
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_categorical(name, **values_cp)
        elif values["type"] == "float":  # corrected this line
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_float(name, **values_cp)

    params["random_state"] = seed_number

    if type(score_func) == str and score_func == "roc_auc":
        score_func = roc_auc_score

    if X_val is not None and y_val is not None:
        # Initialize and train the model
        model = create_pipeline(
            X_train, y_train, model_class(**params), **pipeline_params
        )
        model.fit(X_train, y_train, **fit_params)

        # Predict and evaluate the model
        predictions = model.predict_proba(X_val)[:, 1]
        score = score_func(y_val, predictions)
    else:
        if fit_params != {}:
            raise ValueError("fit_params can only be specified with validation set")
        score = []
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed_number)
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = create_pipeline(
                X_train_fold, y_train_fold, model_class(**params), **pipeline_params
            )
            model.fit(X_train_fold, y_train_fold)
            predictions = model.predict_proba(X_val_fold)[:, 1]
            score.append(score_func(y_val_fold, predictions))
        score = np.mean(score)

    return score


def optimize_model(
    model_class: BaseEstimator,
    param_space: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    cv: int = 5,
    pipeline_params: Dict[str, Any] = {},
    fit_params: Dict[str, Any] = {},
    score_func: str = "roc_auc",
    n_trials: int = None,
    timeout: int = None,
    seed_number: int = None,
    n_jobs: int = 1,
) -> Tuple[Dict[str, Any], Pipeline]:
    
    if param_space == "suggest":
        if model_class.__name__ in hyperparam_spaces.keys():
            param_space = hyperparam_spaces[model_class.__name__]
        else:
            raise ValueError("No hyperparameter space provided")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed_number)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            model_class,
            pipeline_params,
            fit_params,
            score_func,
            param_space,
            X_train,
            y_train,
            X_val,
            y_val,
            cv,
            seed_number=seed_number,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=n_jobs,
    )

    # Train model with best hyperparameters
    best_params = study.best_params
    model = create_pipeline(
        X_train,
        y_train,
        model_class(random_state=seed_number, **best_params),
        **pipeline_params,
    )
    model.fit(X_train, y_train)
    return best_params, model


def optimize_model_fast(
    model_class: BaseEstimator,
    param_space: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    pipeline_params: Dict[str, Any] = {},
    fit_params: Dict[str, Any] = {},
    score_func: str = "roc_auc",
    n_trials: int = None,
    timeout: int = None,
    seed_number: int = None,
    n_jobs=1,
) -> Tuple[Dict[str, Any], Pipeline]:

    def objective_fast_(
        trial,
        model_class,
        fit_params,
        score_func,
        param_space,
        X_train,
        y_train,
        X_val,
        y_val,
        seed_number,
    ):
        params = {}
        for name, values in param_space.items():
            if values["type"] == "int":
                values_cp = {n: v for n, v in values.items() if n != "type"}
                params[name] = trial.suggest_int(name, **values_cp)
            elif values["type"] == "categorical":
                values_cp = {n: v for n, v in values.items() if n != "type"}
                params[name] = trial.suggest_categorical(name, **values_cp)
            elif values["type"] == "float":  # corrected this line
                values_cp = {n: v for n, v in values.items() if n != "type"}
                params[name] = trial.suggest_float(name, **values_cp)

        try:
            params["random_state"] = seed_number
            model = model_class(**params)
        except:
            del params["random_state"]
            model = model_class(**params)

        model.fit(X_train, y_train, **fit_params)

        # Predict and evaluate the model
        y_val_score = model.predict_proba(X_val)[:, 1]
        if type(score_func) == str and score_func == "roc_auc":
            score = roc_auc_score(y_val, y_val_score)
        else:
            y_val_pred = y_val_score > ks_threshold(y_val, y_val_score)
            score = score_func(y_val, y_val_pred, y_val_score)

        return score

    if param_space == "suggest":
        if model_class.__name__ in hyperparam_spaces.keys():
            param_space = hyperparam_spaces[model_class.__name__]
        else:
            raise ValueError("No hyperparameter space provided")

    # pipeline to preprocess
    preprocess = create_pipeline(
        X_train,
        y_train,
        None,
        **pipeline_params,
    )
    preprocess.fit(X_train, y_train)
    X_train_preprocessed = preprocess[:-1].transform(X_train)
    X_val_preprocessed = preprocess[:-1].transform(X_val)

    fit_params_wt_pip = fit_params.copy()
    for key in fit_params.keys():
        fit_params_wt_pip[key.split("__")[1]] = fit_params_wt_pip.pop(key)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed_number)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_fast_(
            trial,
            model_class,
            fit_params_wt_pip,
            score_func,
            param_space,
            X_train_preprocessed,
            y_train,
            X_val_preprocessed,
            y_val,
            seed_number=seed_number,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=n_jobs,
    )

    # Train model with best hyperparameters
    best_params = study.best_params
    try:
        best_params["random_state"] = seed_number
        model = create_pipeline(
            X_train,
            y_train,
            model_class(random_state=seed_number, **best_params),
            **pipeline_params,
        )
        model.fit(X_train, y_train, **fit_params)
    except:
        del best_params["random_state"]
        model = create_pipeline(
            X_train,
            y_train,
            model_class(**best_params),
            **pipeline_params,
        )
        model.fit(X_train, y_train, **fit_params)

    return best_params, model


