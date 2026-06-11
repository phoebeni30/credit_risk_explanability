from typing import Dict, Tuple, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
from .config import MODEL_ORDER, MODEL_COLORS


def false_positive_rate(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:

    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn)



def get_metrics(
    name_model_dict: Dict[str, Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> pd.DataFrame:

    models_dict = {}
    for name, model in name_model_dict.items():
        if type(model) == list:
            model_ = model[0]
            threshold_ = model[1]
        else:
            model_ = model
            threshold_ = threshold

        if hasattr(model_, "predict_proba"):
            y_prob = model_.predict_proba(X)[:, 1]
            y_pred = (y_prob >= threshold_).astype("int")
        else:
            y_prob = None
            y_pred = model_.predict(X)

        models_dict[name] = (y_pred, y_prob)

    def get_metrics_df(
        models_dict,
        y_true,
    ):
        metrics_score_dict = {
            "AUC": lambda x: roc_auc_score(y_true, x) if x is not None else None,
            "Brier Score": lambda x: (
                brier_score_loss(y_true, x) if x is not None else None
            ),
        }
        metrics_dict = {
            "Balanced Accuracy": lambda x: balanced_accuracy_score(y_true, x),
            "Accuracy": lambda x: accuracy_score(y_true, x),
            "Precision": lambda x: precision_score(y_true, x, zero_division=0),
            "Recall": lambda x: recall_score(y_true, x),
            "F1": lambda x: f1_score(y_true, x),
        }
        df_dict = {}
        for metric_name, metric_func in metrics_score_dict.items():
            df_dict[metric_name] = [
                metric_func(scores) for (_, scores) in models_dict.values()
            ]
        for metric_name, metric_func in metrics_dict.items():
            df_dict[metric_name] = [
                metric_func(preds) for (preds, _) in models_dict.values()
            ]
        return pd.DataFrame.from_dict(
            df_dict, orient="index", columns=models_dict.keys()
        ).T

    metrics = get_metrics_df(models_dict, y)
    metrics = metrics.reset_index().rename(columns={"index": "model"})
    return metrics



# ================= PLOT CHARTS =====================
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Return all paper metrics: accuracy, precision, recall, F1, ROC-AUC, MCC."""
    return {
        'Accuracy':          round(accuracy_score(y_true, y_pred), 4),
        'Balanced Accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
        'Precision':         round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'Recall':            round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'F1 Score':          round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'ROC-AUC':           round(roc_auc_score(y_true, y_prob), 4),
        'Brier Score':       round(brier_score_loss(y_true, y_prob), 4)
    }
 
 
def evaluate_all(trained: dict, X_test, y_test,
                 threshold: float = 0.5) -> tuple:
    rows, preds, probs = {}, {}, {}
 
    for name, model in trained.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        rows[name]  = compute_metrics(y_test, y_pred, y_prob)
        preds[name] = y_pred
        probs[name] = y_prob
        print(f"  [eval]  {name:25s}  acc={rows[name]['Accuracy']:.4f}  "
              f"auc={rows[name]['ROC-AUC']:.4f}  mcc={rows[name]['MCC']:.4f}")
 
    ordered    = [m for m in MODEL_ORDER if m in rows]
    metrics_df = pd.DataFrame(rows).T.loc[ordered]
    return metrics_df, preds, probs

def save_metrics(metrics_df: pd.DataFrame, filename: str = 'metrics_summary.csv',
                 out_dir: Path = None):
    if out_dir is None:
        out_dir = PATHS['data_processed']
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    metrics_df.to_csv(path)
    print(f"[eval]  metrics saved → {path}")
 
 
# ── Confusion Matrices ──────────────────────────────────────────────────────
 
def plot_confusion_matrix_single(y_true, y_pred, model_name: str,
                                 save_dir: Path = None):
    """Normalised confusion matrix for one model."""
    if save_dir is None:
        save_dir = PATHS['figures']
    save_dir.mkdir(parents=True, exist_ok=True)
 
    cm   = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(f'Normalised Confusion Matrix: {model_name}')
    plt.tight_layout()
 
    safe = model_name.replace(' ', '_')
    path = save_dir / f'cm_{safe}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig]   confusion matrix '{model_name}' → {path}")
 
 
def plot_all_confusion_matrices(preds: dict, y_test, save_dir: Path = None):
    """Save a normalised confusion-matrix PNG for every model."""
    if save_dir is None:
        save_dir = PATHS['figures']
    for name, y_pred in preds.items():
        plot_confusion_matrix_single(y_test, y_pred, name, save_dir)
    print(f"[eval]  all confusion matrices saved")
 
 
# ── ROC Curves ──────────────────────────────────────────────────────────────
 
def plot_roc_curves(probs: dict, y_test, save_dir: Path = None):
    """
    Fig 24: All models on one ROC plot.
    Fig 25: XGBoost standalone ROC (if present).
    """
    if save_dir is None:
        save_dir = PATHS['figures']
    save_dir.mkdir(parents=True, exist_ok=True)
 
    # Combined ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random (AUC=0.5)')
 
    ordered = [m for m in MODEL_ORDER if m in probs]
    for name in ordered:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        auc = roc_auc_score(y_test, probs[name])
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
                color=MODEL_COLORS.get(name))
 
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
 
    path = save_dir / 'fig24_roc_curves.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig]   ROC curves → {path}")

# ── Metric Bar Charts ───────────────────────────────────────────────────────
 
def _bar_chart(metrics_df: pd.DataFrame, metric: str,
               fig_num: int, save_dir: Path):
    ordered = [m for m in MODEL_ORDER if m in metrics_df.index]
    values  = metrics_df.loc[ordered, metric]
    colors  = [MODEL_COLORS.get(m, '#888') for m in ordered]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ordered, values, color=colors, width=0.55, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_xticklabels(ordered, rotation=15, ha='right')
    plt.tight_layout()
 
    path = save_dir / f'fig{fig_num:02d}_{metric.lower().replace(" ", "_")}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig]   {metric} bar chart → {path}")
 
 
def plot_metric_comparisons(metrics_df: pd.DataFrame, save_dir: Path = None):
    """Save bar-chart PNGs for Accuracy, Precision, Recall, F1."""
    if save_dir is None:
        save_dir = PATHS['figures']
    save_dir.mkdir(parents=True, exist_ok=True)
    _bar_chart(metrics_df, 'Accuracy',  20, save_dir)
    _bar_chart(metrics_df, 'Precision', 21, save_dir)
    _bar_chart(metrics_df, 'Recall',    22, save_dir)
    _bar_chart(metrics_df, 'F1 Score',  23, save_dir)
    print(f"[eval]  metric bar charts saved")
 
 
# ────────────────────────────────────────────────────
 
def plot_feature_importance(model, feature_names: list,
                            model_name: str = 'XGBoost',
                            top_k: int = 15,
                            save_dir: Path = None):
    if save_dir is None:
        save_dir = PATHS['figures']
    save_dir.mkdir(parents=True, exist_ok=True)
 
    # Support both raw estimators and sklearn Pipelines
    estimator = model[-1] if hasattr(model, '__getitem__') else model
    importance = estimator.feature_importances_
 
    idx          = np.argsort(importance)[-top_k:]
    top_features = [feature_names[i] for i in idx]
    top_values   = importance[idx]
 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features, top_values * 100, color='#4CAF50')
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title(f'Feature Importance – {model_name}')
    plt.tight_layout()
 
    safe = model_name.replace(' ', '_')
    path = save_dir / f'fig28_feature_importance_{safe}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig]   feature importance → {path}")
 
 
# ────────────────────────────────────────────────────
 
def print_summary(metrics_df: pd.DataFrame, save_csv: bool = True,
                  out_dir: Path = None):
    """Print Table-3-style summary and optionally save CSV."""
    print("\n" + "=" * 75)
    print("SUMMARY OF PERFORMANCE METRICS  (Table 3)")
    print("=" * 75)
    print(metrics_df.to_string())
    print("=" * 75)
 
    best = metrics_df['Accuracy'].idxmax()
    print(f"\nBest model: {best}  "
          f"(accuracy={metrics_df.loc[best,'Accuracy']:.4f}, "
          f"MCC={metrics_df.loc[best,'MCC']:.4f})\n")
 
    if save_csv:
        save_metrics(metrics_df, out_dir=out_dir)
 
 
# ────────────────────────────────────────────────────
 
def run_full_evaluation(trained: dict, X_test, y_test,
                        feature_names: list = None,
                        z_sensitive=None,
                        save_dir: Path = None,
                        out_dir: Path = None):

    metrics_df, preds, probs = evaluate_all(trained, X_test, y_test)
    print_summary(metrics_df, save_csv=True, out_dir=out_dir)
 
    plot_all_confusion_matrices(preds, y_test, save_dir)
    plot_roc_curves(probs, y_test, save_dir)
    plot_metric_comparisons(metrics_df, save_dir)

 
    # Feature importance for tree-based models
    if feature_names is not None:
        for name in ['XGBoost', 'LGBM', 'Random Forest']:
            if name in trained:
                try:
                    plot_feature_importance(trained[name], feature_names,
                                            model_name=name,
                                            save_dir=save_dir)
                except Exception as e:
                    print(f"  [warn]  feature importance skipped for {name}: {e}")

    print("\n[eval]  full evaluation complete.")