from typing import Optional, Tuple, List, Dict, Union, Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import numpy as np


class EBEEncode(
    TransformerMixin,
    BaseEstimator,
):
    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EBEEncode":
        self.feature_names_in_ = []
        self.n_features, self.n_items = X.shape[1], X.shape[0]
        self._aux_dict_main = {}
        self.mean = {}
        self.unique_values = {col: X[col].unique() for col in X.columns}

        for i in range(self.n_features):
            Xi = X.iloc[:, i]
            X_name = X.iloc[:, i].name

            y = pd.Series(y, index=X.index)
            aux_dict = pd.Series(y).groupby(Xi).agg(["mean", "count"]).to_dict()
            self._aux_dict_main[X_name] = aux_dict
            self.feature_names_in_.append(X_name)
            self.mean[X_name] = y.mean()
        
        return self

    def transform(self, X: pd.DataFrame, y : Optional[pd.Series] = None) -> np.ndarray:

        Xt_list = []
        for X_name in self.feature_names_in_:
            Xi = X[X_name].copy()
            
            means_dict = self._aux_dict_main[X_name]["mean"]
            counts_dict = self._aux_dict_main[X_name]["count"]
            global_mean = self.mean[X_name]
            
            # Handle unknown categories: Using the mode of the means as default fallback
            default_mean = pd.Series(list(means_dict.values())).mode()[0]
            group_mean = Xi.map(means_dict).fillna(default_mean).astype(float)
            group_count = Xi.map(counts_dict).fillna(0).astype(float)

            # 2. Apply the smoothing formula
            # Formula: (mean * count + k * global_mean) / (count + k)
            Xt = (
                (group_mean * group_count + self.k * global_mean)
                / (group_count + self.k)
            )
            
            Xt_list.append(Xt.values.reshape(-1, 1))
            
        return np.hstack(Xt_list)

    def get_feature_names_out(
        self, input_features: Union[pd.DataFrame, List]
    ) -> List[str]:
        
        if isinstance(input_features, pd.DataFrame):
            return [col for col in input_features.columns]
        else:
            return [col for col in input_features]


def create_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    classifier: Optional[BaseEstimator] = None,
    cat_cols: Union[str, List[str]] = "infer",
    onehot: bool = True,
    onehotdrop: bool = False,
    normalize: bool = True,
    do_EBE: bool = True,
    crit: int = 3,
):
    if cat_cols == "infer":
        num_cols = [
            col for col in X.columns if X[col].dtype.kind in ["b", "i", "u", "f", "c"]
        ]
        cat_cols = [col for col in X.columns if X[col].dtype.kind in ["O", "S", "U"]]
        ebe_cols = [col for col in cat_cols if X[col].nunique() >= crit if do_EBE]
        cat_cols = [item for item in cat_cols if item not in ebe_cols]

    else:
        num_cols = X.columns.difference(cat_cols).tolist()
        ebe_cols = [col for col in cat_cols if X[col].nunique() >= crit if do_EBE]
        cat_cols = [item for item in cat_cols if item not in ebe_cols]

    # ========= PIPELINE =========
    # 1 fill - Fill NaNs: num_cols, cat_cols, ebe_cols
    numeric_nan_fill_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="mean"))]
    )
    categorical_nan_fill_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    fill_pipe = ColumnTransformer(
        transformers=[
            ("num", numeric_nan_fill_transformer, num_cols),
            ("cat", categorical_nan_fill_transformer, cat_cols),
            ("ebe", categorical_nan_fill_transformer, ebe_cols),
        ],
        verbose_feature_names_out=False,
    )
    fill_pipe.set_output(transform="pandas")

    # 2: le - Encode categorical: num_cols -> passthrough, cat_cols -> ordinal (unknown = -1), do_ebe -> true: similar to num | do_ebe -> false: similar to ordinal (unknown = -1)
    encoder_pipe = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            (
                "cat",
                OrdinalEncoder(
                    dtype=np.int64,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            (
                "ebe",
                (
                    EBEEncode()
                    if do_EBE
                    else OrdinalEncoder(
                        dtype=np.int64,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    )
                ),
                ebe_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )
    encoder_pipe.set_output(transform="pandas")

    # 3 ss - Scalling
    scaling_pipe = ColumnTransformer(
        [
            ("scaler", StandardScaler(), num_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    scaling_pipe.set_output(transform="pandas")

    # 4 ohot - Onehot encoder: Apply ONLY on cat_cols
    onehot_pipe = ColumnTransformer(
        transformers=[
            (
                "onehot_encoder",
                (
                    OneHotEncoder(
                        drop="if_binary", sparse_output=False, handle_unknown="ignore"
                    )
                    if onehotdrop
                    else OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                ),
                cat_cols,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    onehot_pipe.set_output(transform="pandas")

    # Combine all the transformers in a single pipeline
    pipeline = Pipeline(
        steps=[
            ("fill", fill_pipe),
            ("le", encoder_pipe),
            ("ss", scaling_pipe if normalize else None),
            ("ohot", onehot_pipe if onehot else None),
            ("classifier", classifier),
        ],
    )

    return pipeline


def ks_threshold(
    y_true: Union[np.ndarray, pd.Series], y_score: Union[np.ndarray, pd.Series]
) -> float:

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    return opt_threshold