"""Metrics, operating-point selection, calibration, fairness, and figures."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import CAPACITY_FRACTION, FIGURE_DIR, REPORT_DIR


# --------------------------------------------------------------------------- #
# Operating point
# --------------------------------------------------------------------------- #
def capacity_threshold(scores: np.ndarray, fraction: float = CAPACITY_FRACTION) -> float:
    """Threshold that flags the top `fraction` of patients by risk score.

    A follow-up team can call a fixed number of patients per day, so the useful
    operating point is a capacity budget, not an arbitrary 0.5 cutoff.
    """
    return float(np.quantile(scores, 1 - fraction))


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    flagged = float(y_pred.mean())
    base_rate = float(y_true.mean())
    precision = precision_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "flagged_fraction": flagged,
        "precision": float(precision),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        # How much better than calling a random 20% of patients.
        "lift_over_base_rate": float(precision / base_rate) if base_rate else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Threshold-free discrimination + calibration + the chosen operating point."""
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "base_rate": float(np.mean(y_true)),
        "at_capacity": metrics_at_threshold(y_true, scores, threshold),
        "at_0.5": metrics_at_threshold(y_true, scores, 0.5),
    }


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
def subgroup_report(
    y_true: np.ndarray, scores: np.ndarray, groups: pd.Series, threshold: float, min_n: int = 200
) -> pd.DataFrame:
    """Per-subgroup flag rate, recall, and precision at the deployed threshold.

    A model can look fine overall while systematically under-flagging a subgroup;
    this is the check that has to pass before anything like this reaches a ward.
    """
    rows = []
    y_pred = (scores >= threshold).astype(int)
    for value, idx in groups.groupby(groups).groups.items():
        mask = groups.index.isin(idx)
        if mask.sum() < min_n:
            continue
        yt, yp, sc = y_true[mask], y_pred[mask], scores[mask]
        rows.append(
            {
                "group": value,
                "n": int(mask.sum()),
                "base_rate": float(yt.mean()),
                "flag_rate": float(yp.mean()),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "roc_auc": float(roc_auc_score(yt, sc)) if len(np.unique(yt)) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> None:
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path.relative_to(REPORT_DIR.parent)}")


def plot_roc_curves(y_true: np.ndarray, score_map: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, scores):.3f})", lw=1.6)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — test set")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "roc_curves.png")


def plot_pr_curves(y_true: np.ndarray, score_map: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_true, scores):.3f})", lw=1.6)
    ax.axhline(y_true.mean(), ls="--", c="k", lw=0.8, label=f"base rate={y_true.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall — test set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "pr_curves.png")


def plot_calibration(y_true: np.ndarray, score_map: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, scores in score_map.items():
        prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", lw=1.4, ms=4,
                label=f"{name} (Brier={brier_score_loss(y_true, scores):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.set(xlabel="Predicted risk", ylabel="Observed readmission rate", title="Calibration — test set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "calibration.png")


def plot_capacity_sweep(y_true: np.ndarray, scores: np.ndarray, model_name: str) -> None:
    """Precision / recall as a function of how many patients the team can call."""
    fractions = np.arange(0.05, 0.55, 0.01)
    precisions, recalls = [], []
    for frac in fractions:
        thr = capacity_threshold(scores, frac)
        m = metrics_at_threshold(y_true, scores, thr)
        precisions.append(m["precision"])
        recalls.append(m["recall"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fractions * 100, np.array(recalls) * 100, label="Recall (of true readmissions caught)", lw=1.8)
    ax.plot(fractions * 100, np.array(precisions) * 100, label="Precision (of flagged patients)", lw=1.8)
    ax.axhline(y_true.mean() * 100, ls="--", c="k", lw=0.8, label="precision if flagging at random")
    ax.axvline(CAPACITY_FRACTION * 100, ls=":", c="gray", lw=1.2, label="chosen capacity")
    ax.set(
        xlabel="% of discharges flagged for follow-up",
        ylabel="%",
        title=f"Follow-up capacity trade-off — {model_name}",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "capacity_sweep.png")


def plot_confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float, model_name: str) -> None:
    cm = confusion_matrix(y_true, (scores >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["not flagged", "flagged"], yticklabels=["no readmit", "readmit <30d"],
        xlabel="Prediction", ylabel="Actual",
        title=f"{model_name} @ top {CAPACITY_FRACTION:.0%} risk",
    )
    _save(fig, "confusion_matrix.png")


def plot_model_comparison(comparison: pd.DataFrame, base_rate: float) -> None:
    """Bars start at the no-skill value for each metric, not at zero.

    A zero-based axis makes 0.67 and 0.68 look identical. Anchoring ROC-AUC at
    0.5 and PR-AUC at the base rate shows the part of the score that is actually
    earned, without the distortion of an arbitrary truncated axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    floors = {"roc_auc": 0.5, "pr_auc": base_rate}
    labels = {"roc_auc": "random ranking = 0.5", "pr_auc": f"base rate = {base_rate:.3f}"}

    for ax, metric, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC"]):
        data = comparison.sort_values(f"test_{metric}")
        floor = floors[metric]
        ax.barh(data["model"], data[f"test_{metric}"] - floor, left=floor, color="#4c72b0")
        if f"cv_{metric}_mean" in data:
            ax.errorbar(
                data[f"cv_{metric}_mean"], range(len(data)),
                xerr=data[f"cv_{metric}_std"], fmt="o", color="#dd8452", ms=5,
                label="5-fold grouped CV", lw=1.2,
            )
        ax.axvline(floor, color="k", ls="--", lw=0.9, label=labels[metric])
        top = data[f"test_{metric}"].max()
        ax.set_xlim(floor - (top - floor) * 0.08, top + (top - floor) * 0.15)
        ax.set_title(f"{title} (bars = held-out test)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    _save(fig, "model_comparison.png")


def save_json(obj: dict, name: str) -> None:
    path = REPORT_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[report] {path.relative_to(REPORT_DIR.parent)}")
