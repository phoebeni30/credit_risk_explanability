from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PATHS = {
    'data_raw':        BASE_DIR / 'data' / 'raw',
    'data_processed':  BASE_DIR / 'data' / 'processed',
    'figures':         BASE_DIR / 'outputs' / 'figures',
    'models':          BASE_DIR / 'outputs' / 'models',
}

RANDOM_STATE      = 42
TEST_SIZE         = 0.30
PERFORMANCE_WINDOW = 12        # months

hyperparam_spaces = {
    "LogisticRegression": {
        "C": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "tol": {"low": 1e-6, "high": 1e-3, "log": True, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "max_iter": {"low": 1000, "high": 1000, "step": 1, "type": "int"},
        "penalty": {"choices": ["l1", "l2"], "type": "categorical"},
        "solver": {"choices": ["liblinear"], "type": "categorical"},
    },
    "RandomForestClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "max_depth": {"low": 2, "high": 12, "type": "int"},
        "criterion": {"choices": ["gini", "entropy"], "type": "categorical"},
        "min_samples_split": {"low": 2, "high": 256, "type": "int"},
        "min_samples_leaf": {"low": 1, "high": 256, "type": "int"},
        "max_features": {"low": 0.1, "high": 1.0, "type": "float"},
        "class_weight": {"choices": ["balanced"], "type": "categorical"},
    },
    "LGBMClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "learning_rate": {"low": 0.05, "high": 1.0, "type": "float"},
        "num_leaves": {"low": 4, "high": 256, "type": "int"},
        "max_depth": {"low": 2, "high": 12, "type": "int"},
        "min_child_samples": {"low": 1, "high": 256, "type": "int"},
        "colsample_bytree": {"low": 0.1, "high": 1.0, "type": "float"},
        "reg_alpha": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "reg_lambda": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "verbose": {"choices": [-1], "type": "categorical"},
    },
    "MLPClassifier": {
        "hidden_layer_sizes": {
            "choices": [
                [32],
                [64],
                [128],
                [32, 16],
                [64, 32],
                [64, 32, 16],
            ],
            "type": "categorical",
        },
        "learning_rate_init": {"low": 0.001, "high": 0.1, "type": "float", "log": True},
        #"learning_rate_decay_rate": {
        #    "low": 0.1,
        #    "high": 1,
        #    "type": "float",
        #},
        "learning_rate": {
            "choices": ["constant", "invscaling", "adaptive"], 
            "type": "categorical"
        },
        "power_t": {
            "low": 0.1,
            "high": 1.0,
            "type": "float",
        },
        "alpha": {"low": 1e-3, "high": 1e3, "type": "float", "log": True},
        "max_iter": {"low": 10, "high": 100, "type": "int", "step": 10},
        #"class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "batch_size": {"low": 128, "high": 128, "type": "int"},
    },

    "AdaBoostClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "learning_rate": {"low": 0.05, "high": 1.0, "type": "float"},
        #"algorithm": {"choices": ["SAMME"], "type": "categorical"},
    },

    "GradientBoostingClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "learning_rate": {"low": 0.05, "high": 1.0, "type": "float"},
        "max_depth": {"low": 2, "high": 12, "type": "int"},
        "min_samples_split": {"low": 2, "high": 256, "type": "int"},
        "min_samples_leaf": {"low": 1, "high": 256, "type": "int"},
        "max_features": {"low": 0.1, "high": 1.0, "type": "float"},
        "subsample": {"low": 0.5, "high": 1.0, "type": "float"},
    },

    "XGBClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "learning_rate": {"low": 0.05, "high": 1.0, "type": "float"},
        "max_depth": {"low": 2, "high": 12, "type": "int"},
        "min_child_weight": {"low": 1, "high": 256, "type": "int"},
        "colsample_bytree": {"low": 0.1, "high": 1.0, "type": "float"},
        "subsample": {"low": 0.5, "high": 1.0, "type": "float"},
        "reg_alpha": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "reg_lambda": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "scale_pos_weight": {"choices": [1, None], "type": "categorical"},
    },
}   


# Display order matching the paper's bar charts
MODEL_ORDER = [
    'LogisticRegression',
    'MLPClassifier',
    'AdaBoostClassifier',
    'RandomForestClassifier',
    'LGBMClassifier',
    'XGBClassifier',
    'GradientBoostingClassifier'
]

MODEL_COLORS = {
    'LogisticRegression':          '#4472C4',
    'MLPClassifier':               '#ED7D31',
    'AdaBoostClassifier':          '#70AD47',
    'RandomForestClassifier':      '#FFC000',
    'LGBMClassifier':              '#9E480E',
    'XGBClassifier':               '#833C00',
    'GradientBoostingClassifier':  '#36F984',
}
