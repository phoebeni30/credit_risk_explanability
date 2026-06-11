import joblib
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier  # Use default NN from sklearn (Not use nn_models.py yet)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .training import create_pipeline


from src.config import hyperparam_spaces, PATHS

def _model_path(name: str, out_dir) -> Path:
    return Path(out_dir) / f"{name.replace(' ', '_')}.pkl"

def _params_path(name: str, out_dir) -> Path:
    return Path(out_dir) / f"{name.replace(' ', '_')}_params.json"

def get_models() -> dict:
    models = {
            'Logistic Regression': LogisticRegression(),
            'Neural Network':      MLPClassifier(),
            'AdaBoost':            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1)),
            'Random Forest':       RandomForestClassifier(),
            'LGBM':                LGBMClassifier(),
            'XGBoost':             XGBClassifier(),
            'GradientBoosting':    GradientBoostingClassifier(),
        }
    
    return models

def train_all(
    models: dict,
    X_train,
    y_train,
    cat_cols="infer",
    onehot: bool = True,
    onehotdrop: bool = False,
    normalize: bool = True,
    do_EBE: bool = True,
    crit: int = 3,
) -> dict:
    """Wrap every model in create_pipeline, fit, and return trained dict."""
    trained = {}
    for name, model in models.items():
        print(f"  [train] {name} ...", end=' ', flush=True)
        pipeline = create_pipeline(
            X=X_train,
            y=y_train,
            classifier=model,
            cat_cols=cat_cols,
            onehot=onehot,
            onehotdrop=onehotdrop,
            normalize=normalize,
            do_EBE=do_EBE,
            crit=crit,
        )
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        print("done")
    return trained



def save_models(trained: dict, out_dir: Path = None):
    if out_dir is None:
        out_dir = PATHS['models']
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in trained.items():
        path = _model_path(name, out_dir)
        joblib.dump(model, path)
        print(f"  [save]  {name:25s} → {path}")

    print(f"[models]  all saved to {out_dir}/")


def save_best_params(best_params: dict, out_dir: Path = None):
    if out_dir is None:
        out_dir = PATHS['models']
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, params in best_params.items():
        path = _params_path(name, out_dir)
        with open(path, 'w') as f:
            json.dump(params, f, indent=2, default=str)
        print(f"  [params] {name:25s} → {path}")

    print(f"[models]  best params saved to {out_dir}/")


def save_pipeline(name: str, pipeline: Pipeline, best_params: dict = None,
                  out_dir: Path = None):
    if out_dir is None:
        out_dir = PATHS['models']
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = _model_path(name, out_dir)
    joblib.dump(pipeline, pkl_path)
    print(f"  [save]  pipeline '{name}' → {pkl_path}")

    if best_params is not None:
        save_best_params({name: best_params}, out_dir)



def load_models(names: list, out_dir: Path = None) -> dict:
    if out_dir is None:
        out_dir = PATHS['models']

    loaded = {}
    for name in names:
        path = _model_path(name, out_dir)
        loaded[name] = joblib.load(path)
        print(f"  [load]  {name:25s} ← {path}")
    return loaded


def load_best_params(names: list, out_dir: Path = None) -> dict:
    if out_dir is None:
        out_dir = PATHS['models']

    params = {}
    for name in names:
        path = _params_path(name, out_dir)
        with open(path, 'r') as f:
            params[name] = json.load(f)
        print(f"  [params] {name:25s} ← {path}")
    return params