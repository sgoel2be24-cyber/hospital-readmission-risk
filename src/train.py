"""End-to-end training run: prepare data, compare models, calibrate, report.

    python -m src.train

Produces models/best_model.joblib plus everything under reports/.
"""

from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import evaluate as ev
from .config import (
    AGE_BINS,
    AGE_LABELS,
    CAPACITY_FRACTION,
    CV_FOLDS,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from .data_prep import build_dataset, split_feature_types
from .pipeline import assemble, build_models

# numpy 2.x + the macOS Accelerate BLAS emit spurious overflow warnings from
# matmul during MLP training and Platt scaling; they do not affect the results.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.calibration")
# LightGBM's sklearn wrapper invents placeholder feature names when fitted on an
# array, so sklearn warns on every predict. Cosmetic — column order is fixed by
# the shared ColumnTransformer.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Two calibrators are compared. Isotonic usually wins on Brier score but ties
# many scores together, which costs ranking quality. Prefer the better-ranking
# calibrator whenever its Brier score is within this distance of the best.
BRIER_TOLERANCE = 0.002


def grouped_split(X, y, groups, test_size, seed=RANDOM_STATE):
    """Split so that no patient appears on both sides.

    Splitting by row would put different encounters of the same patient into
    train and test; the model then partly memorises patients and the score is
    optimistic. `patient_nbr` is the grouping key.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    left_idx, right_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[left_idx], X.iloc[right_idx],
        y.iloc[left_idx], y.iloc[right_idx],
        groups.iloc[left_idx], groups.iloc[right_idx],
    )


def cross_validate_grouped(model_spec, numeric, categorical, X, y, groups) -> dict:
    """GroupKFold CV — the model-selection estimate, computed on train only."""
    cv = GroupKFold(n_splits=CV_FOLDS)
    roc, pr = [], []
    for train_idx, val_idx in cv.split(X, y, groups):
        pipe = assemble(model_spec, numeric, categorical)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        roc.append(roc_auc_score(y.iloc[val_idx], scores))
        pr.append(average_precision_score(y.iloc[val_idx], scores))
    return {
        "cv_roc_auc_mean": float(np.mean(roc)), "cv_roc_auc_std": float(np.std(roc)),
        "cv_pr_auc_mean": float(np.mean(pr)), "cv_pr_auc_std": float(np.std(pr)),
    }


def main(run_cv: bool = True) -> dict:
    t_start = time.time()
    X, y, groups = build_dataset()
    numeric, categorical = split_feature_types(X)

    # patient-level: full -> (train_full, test), then train_full -> (train, val)
    X_full, X_test, y_full, y_test, g_full, g_test = grouped_split(X, y, groups, TEST_SIZE)
    X_train, X_val, y_train, y_val, g_train, g_val = grouped_split(X_full, y_full, g_full, VAL_SIZE)

    assert not set(g_train) & set(g_test), "patient leaked between train and test"
    assert not set(g_val) & set(g_test), "patient leaked between val and test"
    print(
        f"[split] train={len(X_train):,} ({y_train.mean():.2%} pos) "
        f"val={len(X_val):,} ({y_val.mean():.2%}) test={len(X_test):,} ({y_test.mean():.2%}) "
        f"| patients {g_train.nunique():,}/{g_val.nunique():,}/{g_test.nunique():,}"
    )

    model_specs = build_models(y_train.to_numpy())
    rows, test_scores, val_scores, fitted = [], {}, {}, {}

    for name, spec in model_specs.items():
        t0 = time.time()
        row = {"model": name}

        if run_cv:
            row.update(cross_validate_grouped(spec, numeric, categorical, X_full, y_full, g_full))

        pipe = assemble(spec, numeric, categorical)
        pipe.fit(X_train, y_train)
        val_scores[name] = pipe.predict_proba(X_val)[:, 1]
        test_scores[name] = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe

        # Threshold comes from validation, never from test.
        thr = ev.capacity_threshold(val_scores[name], CAPACITY_FRACTION)
        test_metrics = ev.evaluate(y_test.to_numpy(), test_scores[name], thr)
        row.update(
            {
                "val_roc_auc": roc_auc_score(y_val, val_scores[name]),
                "val_pr_auc": average_precision_score(y_val, val_scores[name]),
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_brier": test_metrics["brier"],
                "test_recall_at_capacity": test_metrics["at_capacity"]["recall"],
                "test_precision_at_capacity": test_metrics["at_capacity"]["precision"],
                "test_lift_at_capacity": test_metrics["at_capacity"]["lift_over_base_rate"],
                "fit_seconds": round(time.time() - t0, 1),
            }
        )
        rows.append(row)
        print(
            f"[model] {name:<20} test ROC-AUC={row['test_roc_auc']:.4f} "
            f"PR-AUC={row['test_pr_auc']:.4f} recall@{CAPACITY_FRACTION:.0%}="
            f"{row['test_recall_at_capacity']:.3f} ({row['fit_seconds']}s)"
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # Selection on validation PR-AUC — test stays untouched until the winner is fixed.
    candidates = comparison[comparison["model"] != "baseline_majority"]
    best_name = candidates.loc[candidates["val_pr_auc"].idxmax(), "model"]
    best_pipe = fitted[best_name]
    print(f"\n[select] best model by validation PR-AUC: {best_name}")

    # --- Calibration ------------------------------------------------------- #
    # Ranking is enough to sort a queue, but a risk *score* a clinician reads as
    # "18% chance" has to actually mean it — and class weighting deliberately
    # distorts the raw probabilities. Both calibrators are fitted on validation
    # and the one with the better validation Brier score wins.
    calibrators, cal_choice = {}, {}
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(FrozenEstimator(best_pipe), method=method)
        cal.fit(X_val, y_val)
        val_p = cal.predict_proba(X_val)[:, 1]
        calibrators[method] = cal
        cal_choice[method] = {
            "val_brier": float(brier_score_loss(y_val, val_p)),
            "val_pr_auc": float(average_precision_score(y_val, val_p)),
        }
        print(f"[calibrate] {method:<9} val Brier={cal_choice[method]['val_brier']:.4f} "
              f"val PR-AUC={cal_choice[method]['val_pr_auc']:.4f}")

    best_brier = min(c["val_brier"] for c in cal_choice.values())
    close_enough = [m for m, c in cal_choice.items() if c["val_brier"] <= best_brier + BRIER_TOLERANCE]
    cal_method = max(close_enough, key=lambda m: cal_choice[m]["val_pr_auc"])
    calibrated = calibrators[cal_method]
    cal_val = calibrated.predict_proba(X_val)[:, 1]
    cal_test = calibrated.predict_proba(X_test)[:, 1]
    thr_cal = ev.capacity_threshold(cal_val, CAPACITY_FRACTION)

    raw_metrics = ev.evaluate(
        y_test.to_numpy(), test_scores[best_name],
        ev.capacity_threshold(val_scores[best_name], CAPACITY_FRACTION),
    )
    cal_metrics = ev.evaluate(y_test.to_numpy(), cal_test, thr_cal)
    print(
        f"[calibrate] chose {cal_method}: test Brier {raw_metrics['brier']:.4f} -> "
        f"{cal_metrics['brier']:.4f}  (PR-AUC {raw_metrics['pr_auc']:.4f} -> {cal_metrics['pr_auc']:.4f})"
    )

    # --- Fairness ---------------------------------------------------------- #
    fairness = {}
    for attribute in ("race", "gender", "age_mid"):
        col = X_test[attribute]
        if attribute == "age_mid":
            col = pd.cut(col, bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
        report = ev.subgroup_report(y_test.to_numpy(), cal_test, col, thr_cal)
        report.insert(0, "attribute", attribute)
        fairness[attribute] = report
        print(f"\n[fairness] {attribute}\n{report.to_string(index=False)}")
    pd.concat(fairness.values()).to_csv(REPORT_DIR / "fairness_report.csv", index=False)

    # --- Figures ----------------------------------------------------------- #
    # The dummy baseline emits a constant score; it has no curve to draw.
    curves = {k: v for k, v in test_scores.items() if k != "baseline_majority"}
    ev.plot_roc_curves(y_test.to_numpy(), curves)
    ev.plot_pr_curves(y_test.to_numpy(), curves)
    ev.plot_calibration(
        y_test.to_numpy(),
        {**curves, f"{best_name} ({cal_method})": cal_test},
    )
    ev.plot_capacity_sweep(y_test.to_numpy(), cal_test, f"{best_name} (calibrated)")
    ev.plot_confusion(y_test.to_numpy(), cal_test, thr_cal, best_name)
    ev.plot_model_comparison(candidates, base_rate=float(y_test.mean()))

    # --- Persist ----------------------------------------------------------- #
    artefact = {
        "model": calibrated,
        "threshold": thr_cal,
        "capacity_fraction": CAPACITY_FRACTION,
        "model_name": best_name,
        "calibration": cal_method,
        "feature_columns": list(X.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
    }
    joblib.dump(artefact, MODEL_DIR / "best_model.joblib")
    joblib.dump(fitted, MODEL_DIR / "all_models.joblib")
    print(f"[save] models/best_model.joblib ({best_name}, threshold={thr_cal:.4f})")

    summary = {
        "dataset": {
            "rows_modelled": int(len(X)),
            "n_features_raw": int(X.shape[1]),
            "n_patients": int(groups.nunique()),
            "positive_rate": float(y.mean()),
            "split": {
                "train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test)),
                "strategy": "GroupShuffleSplit on patient_nbr",
            },
        },
        "best_model": best_name,
        "calibration": {"method_chosen": cal_method, "validation_scores": cal_choice},
        "capacity_fraction": CAPACITY_FRACTION,
        "deployed_threshold": thr_cal,
        "test_metrics_uncalibrated": raw_metrics,
        "test_metrics_calibrated": cal_metrics,
        "model_comparison": comparison.to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    ev.save_json(summary, "metrics.json")
    print(f"\n[done] {summary['runtime_seconds']}s")
    return summary


if __name__ == "__main__":
    main()
