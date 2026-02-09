"""Utilidades de treinamento, avaliação e visualização de modelos."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve
from src.config import FIGURES_DIR, RANDOM_SEED


def evaluate_binary_proba(y_true, y_prob, verbose=True):
    """Calcula métricas de avaliação para probabilidades binárias.

    Returns:
        dict com AUC-ROC, Gini, KS, Brier Score, PR-AUC, Log Loss
    """
    auc_roc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc_roc - 1
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # KS Statistic
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)

    metrics = {
        "AUC-ROC": auc_roc,
        "Gini": gini,
        "KS": ks,
        "Brier Score": brier,
        "PR-AUC": pr_auc,
        "Log Loss": logloss,
    }

    if verbose:
        print("=" * 40)
        for name, val in metrics.items():
            print(f"  {name:<15}: {val:.4f}")
        print("=" * 40)

    return metrics


def plot_calibration_curve(y_true, y_prob, n_bins=10, title="Curva de Calibração", save_path=None):
    """Plota curva de calibração (probabilidade predita vs taxa real)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    ax.plot(prob_pred, prob_true, "s-", label="Modelo", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", label="Calibração perfeita")
    ax.set_xlabel("Probabilidade Predita (média por bin)")
    ax.set_ylabel("Taxa Real de Default")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_ks_curve(y_true, y_prob, title="Curva KS", save_path=None):
    """Plota curva KS (separação entre distribuições)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_stat = np.max(tpr - fpr)
    ks_idx = np.argmax(tpr - fpr)

    ax.plot(thresholds[1:], tpr[1:], label="TPR (Sensibilidade)", color="steelblue")
    ax.plot(thresholds[1:], fpr[1:], label="FPR (1 - Especificidade)", color="coral")
    ax.axvline(thresholds[ks_idx], color="gray", linestyle="--",
               label=f"KS = {ks_stat:.4f} (threshold={thresholds[ks_idx]:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Taxa")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_roc_pr_curves(y_true, y_prob, title_prefix="", save_path=None):
    """Plota curvas ROC e Precision-Recall lado a lado."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, color="steelblue", label=f"AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"{title_prefix}Curva ROC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    ax2.plot(recall, precision, color="coral", label=f"PR-AUC = {pr_auc:.4f}")
    ax2.axhline(baseline, color="gray", linestyle="--", label=f"Baseline = {baseline:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title_prefix}Curva Precision-Recall")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def temporal_train_val_split(df, train_end, val_start, val_end=None):
    """Split respeitando ordem temporal.

    Args:
        df: DataFrame com coluna SAFRA_REF
        train_end: Último mês de treino (inclusive)
        val_start: Primeiro mês de validação (inclusive)
        val_end: Último mês de validação (inclusive), ou None para pegar tudo

    Returns:
        (df_train, df_val)
    """
    train_end = pd.Timestamp(train_end)
    val_start = pd.Timestamp(val_start)

    df_train = df[df["SAFRA_REF"] <= train_end].copy()

    if val_end is not None:
        val_end = pd.Timestamp(val_end)
        df_val = df[(df["SAFRA_REF"] >= val_start) & (df["SAFRA_REF"] <= val_end)].copy()
    else:
        df_val = df[df["SAFRA_REF"] >= val_start].copy()

    return df_train, df_val


def expanding_window_cv(df, folds_config):
    """Cross-validation com janela expansiva.

    Args:
        df: DataFrame com coluna SAFRA_REF
        folds_config: Lista de dicts com {train_end, val_start, val_end}

    Yields:
        (fold_num, df_train, df_val)
    """
    for i, fold in enumerate(folds_config):
        df_train, df_val = temporal_train_val_split(
            df, fold["train_end"], fold["val_start"], fold.get("val_end")
        )
        yield i + 1, df_train, df_val


# Configuração de folds para expanding window CV
EXPANDING_CV_FOLDS = [
    {"train_end": "2019-06-01", "val_start": "2019-07-01", "val_end": "2019-12-01"},
    {"train_end": "2019-12-01", "val_start": "2020-01-01", "val_end": "2020-06-01"},
    {"train_end": "2020-06-01", "val_start": "2020-07-01", "val_end": "2020-12-01"},
    {"train_end": "2020-12-01", "val_start": "2021-01-01", "val_end": "2021-06-01"},
]


def plot_model_comparison(results_dict, save_path=None):
    """Plota comparação de métricas entre modelos.

    Args:
        results_dict: Dict {model_name: metrics_dict}
    """
    df = pd.DataFrame(results_dict).T
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics_to_plot = ["AUC-ROC", "Gini", "KS", "Brier Score", "PR-AUC", "Log Loss"]
    colors = sns.color_palette("viridis", len(results_dict))

    for ax, metric in zip(axes.flatten(), metrics_to_plot):
        values = df[metric]
        bars = ax.bar(range(len(values)), values, color=colors)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(df.index, rotation=45, ha="right")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Comparação de Modelos", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
