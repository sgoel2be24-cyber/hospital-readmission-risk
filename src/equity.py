"""Two fairness questions the standard subgroup table does not answer.

    python -m src.equity

1. **Stratified thresholds.** A single global threshold flags 23.0% of patients
   aged 80+ while ranking them worst (ROC-AUC 0.613) — the tool spends its most
   capacity where its ordering is least trustworthy. Does giving each age band
   its own threshold help?

   Expect a trade, not a free win: a globally-thresholded calibrated model
   already maximises expected true positives for a fixed number of calls, so
   reallocating capacity can only *reduce* total recall. The question is whether
   the equity gain is worth the efficiency loss — and that is a decision for a
   hospital, not for a model. This quantifies both sides.

2. **Subgroup calibration.** The usual report covers per-group *discrimination*.
   For a tool whose entire output is a probability, per-group *calibration*
   matters at least as much: a model can be well calibrated overall while
   telling a nurse "18%" for a patient whose real risk is 27%.
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

import joblib

from .config import (
    AGE_BINS,
    AGE_LABELS,
    CAPACITY_FRACTION,
    FIGURE_DIR,
    MODEL_DIR,
    OLDEST_BAND,
    REPORT_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from .data_prep import build_dataset
from .evaluate import capacity_threshold
from .train import grouped_split

warnings.filterwarnings("ignore", message="X does not have valid feature names")

MIN_GROUP = 200


def age_band(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)


# --------------------------------------------------------------------------- #
# 1. Stratified thresholds
# --------------------------------------------------------------------------- #
def stratified_thresholds(val_scores, val_bands, fraction) -> dict:
    """One capacity threshold per band, each fitted on validation only."""
    out = {}
    for band in AGE_LABELS:
        mask = (val_bands == band).to_numpy()
        if mask.sum() < MIN_GROUP:
            continue
        out[band] = capacity_threshold(val_scores[mask], fraction)
    return out


def apply_thresholds(scores, bands, thresholds, global_thr) -> np.ndarray:
    pred = np.zeros(len(scores), dtype=int)
    for i, (s, b) in enumerate(zip(scores, bands)):
        pred[i] = int(s >= thresholds.get(b, global_thr))
    return pred


def band_table(y, pred, scores, bands) -> pd.DataFrame:
    rows = []
    for band in AGE_LABELS:
        mask = (bands == band).to_numpy()
        if mask.sum() < MIN_GROUP:
            continue
        rows.append({
            "band": band,
            "n": int(mask.sum()),
            "base_rate": float(y[mask].mean()),
            "flag_rate": float(pred[mask].mean()),
            "recall": float(recall_score(y[mask], pred[mask], zero_division=0)),
            "precision": float(precision_score(y[mask], pred[mask], zero_division=0)),
            "roc_auc": float(roc_auc_score(y[mask], scores[mask])),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Subgroup calibration
# --------------------------------------------------------------------------- #
def calibration_table(y, scores, groups, label) -> pd.DataFrame:
    """Per-group Brier, plus calibration-in-the-large and slope.

    intercept ratio = mean(predicted) / mean(observed):
        >1 the model overstates risk for this group, <1 it understates.
    slope from a logistic fit of the outcome on the logit of the prediction:
        1.0 is ideal; <1 means predictions are too spread out for this group.
    """
    from sklearn.linear_model import LogisticRegression

    rows = []
    for value in sorted(set(groups)):
        mask = (groups == value).to_numpy()
        if mask.sum() < MIN_GROUP:
            continue
        yg, sg = y[mask], np.clip(scores[mask], 1e-6, 1 - 1e-6)
        if len(np.unique(yg)) < 2:
            continue
        logit = np.log(sg / (1 - sg)).reshape(-1, 1)
        slope = float(LogisticRegression(penalty=None, max_iter=1000).fit(logit, yg).coef_[0][0])
        rows.append({
            "attribute": label,
            "group": value,
            "n": int(mask.sum()),
            "observed_rate": float(yg.mean()),
            "predicted_mean": float(sg.mean()),
            "calibration_ratio": float(sg.mean() / yg.mean()) if yg.mean() else np.nan,
            "calibration_slope": slope,
            "brier": float(brier_score_loss(yg, sg)),
        })
    return pd.DataFrame(rows)


def main() -> dict:
    artefact = joblib.load(MODEL_DIR / "best_model.joblib")
    X, y, groups = build_dataset()
    X_full, X_test, y_full, y_test, g_full, _ = grouped_split(X, y, groups, TEST_SIZE)
    _, X_val, _, y_val, _, _ = grouped_split(X_full, y_full, g_full, VAL_SIZE)

    val_scores = artefact["model"].predict_proba(X_val)[:, 1]
    test_scores = artefact["model"].predict_proba(X_test)[:, 1]
    yt = y_test.to_numpy()

    val_bands = age_band(X_val["age_mid"])
    test_bands = age_band(X_test["age_mid"])
    global_thr = artefact["threshold"]

    # ---------------- 1. stratified vs global ------------------------------ #
    thresholds = stratified_thresholds(val_scores, val_bands, CAPACITY_FRACTION)
    print("[thresholds] fitted on validation, one per age band")
    print(f"  global {global_thr:.4f}")
    for b, t in thresholds.items():
        print(f"  {b:<6} {t:.4f}")

    pred_global = (test_scores >= global_thr).astype(int)
    pred_strat = apply_thresholds(test_scores, test_bands, thresholds, global_thr)

    tbl_global = band_table(yt, pred_global, test_scores, test_bands)
    tbl_strat = band_table(yt, pred_strat, test_scores, test_bands)
    tbl_global.insert(0, "policy", "global")
    tbl_strat.insert(0, "policy", "age-stratified")
    comparison = pd.concat([tbl_global, tbl_strat], ignore_index=True)

    overall = {}
    for name, pred in (("global", pred_global), ("age-stratified", pred_strat)):
        overall[name] = {
            "flagged_fraction": float(pred.mean()),
            "recall": float(recall_score(yt, pred, zero_division=0)),
            "precision": float(precision_score(yt, pred, zero_division=0)),
            "true_positives": int(((pred == 1) & (yt == 1)).sum()),
        }

    print("\n[per-band comparison]")
    print(comparison.round(4).to_string(index=False))
    print("\n[overall]")
    for k, v in overall.items():
        print(f"  {k:<15} flagged {v['flagged_fraction']:.3f}  recall {v['recall']:.4f}  "
              f"precision {v['precision']:.4f}  TP {v['true_positives']}")

    delta_tp = overall["age-stratified"]["true_positives"] - overall["global"]["true_positives"]
    g_old = tbl_global[tbl_global["band"] == OLDEST_BAND].iloc[0]
    s_old = tbl_strat[tbl_strat["band"] == OLDEST_BAND].iloc[0]
    print(f"\n[verdict] stratifying costs {-delta_tp} true positives overall; "
          f"{OLDEST_BAND} flag rate {g_old['flag_rate']:.3f} -> {s_old['flag_rate']:.3f}, "
          f"precision {g_old['precision']:.3f} -> {s_old['precision']:.3f}")

    # ---------------- 2. subgroup calibration ------------------------------ #
    cal_frames = [
        calibration_table(yt, test_scores, X_test["race"], "race"),
        calibration_table(yt, test_scores, X_test["gender"], "gender"),
        calibration_table(yt, test_scores, test_bands, "age_band"),
    ]
    cal = pd.concat(cal_frames, ignore_index=True)
    print("\n[subgroup calibration]")
    print(cal.round(4).to_string(index=False))

    # ---------------- figures ---------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    width = 0.36
    xs = np.arange(len(tbl_global))
    ax.bar(xs - width / 2, tbl_global["flag_rate"], width, label="global threshold", color="#4c72b0")
    ax.bar(xs + width / 2, tbl_strat["flag_rate"], width, label="age-stratified", color="#dd8452")
    ax.axhline(CAPACITY_FRACTION, ls="--", c="k", lw=0.9, label="capacity budget")
    ax.set_xticks(xs)
    ax.set_xticklabels(tbl_global["band"])
    ax.set(xlabel="age band", ylabel="fraction flagged", title="Where the follow-up capacity goes")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for band in AGE_LABELS:
        mask = (test_bands == band).to_numpy()
        if mask.sum() < MIN_GROUP:
            continue
        pt, pp = calibration_curve(yt[mask], test_scores[mask], n_bins=6, strategy="quantile")
        ax.plot(pp, pt, "o-", lw=1.5, ms=4, label=f"{band} (n={mask.sum():,})")
    ax.plot([0, 0.45], [0, 0.45], "k--", lw=0.9, label="perfect")
    ax.set(xlabel="predicted risk", ylabel="observed rate", title="Calibration within each age band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    comparison.to_csv(REPORT_DIR / "threshold_policy_comparison.csv", index=False)
    cal.to_csv(REPORT_DIR / "subgroup_calibration.csv", index=False)
    summary = {
        "capacity_fraction": CAPACITY_FRACTION,
        "global_threshold": float(global_thr),
        "stratified_thresholds": {k: float(v) for k, v in thresholds.items()},
        "overall": overall,
        "true_positives_delta": int(delta_tp),
        "per_band": comparison.to_dict(orient="records"),
        "subgroup_calibration": cal.to_dict(orient="records"),
    }
    (REPORT_DIR / "equity.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n[report] reports/equity.json · threshold_policy_comparison.csv · "
          "subgroup_calibration.csv · figures/equity.png")
    return summary


if __name__ == "__main__":
    main()
