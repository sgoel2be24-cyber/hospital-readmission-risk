"""Ablation study: does each proposed change actually earn its place?

    python -m src.experiments

Every variant is scored with the same GroupKFold CV over the *training* patients.
The test set is never read here — that is what keeps the final number in
`src/train.py` an honest estimate rather than the best of a dozen peeks.

A change is adopted only if it improves mean CV PR-AUC without hurting mean CV
ROC-AUC, and the gain is reported next to the fold-to-fold standard deviation so
noise is not mistaken for improvement.
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import CV_FOLDS, RANDOM_STATE, REPORT_DIR, TEST_SIZE
from .data_prep import EXTRA_FEATURES, build_dataset, split_feature_types
from .pipeline import assemble, build_models
from .train import grouped_split

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Variant builders
# --------------------------------------------------------------------------- #
def native_categorical_pipeline(numeric: list[str], categorical: list[str], y_train) -> Pipeline:
    """LightGBM splitting on categories directly instead of on one-hot columns.

    One-hot encoding forces the tree to isolate one level per split. Native
    handling lets a single split separate any subset of levels, which matters
    most for the wide columns here (medical_specialty, payer_code, diagnosis
    groups). Requires ordinal codes plus an explicit categorical_feature list.
    """
    prep = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # After the ColumnTransformer, categoricals occupy the trailing positions.
    categorical_positions = list(range(len(numeric), len(numeric) + len(categorical)))
    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.0113,
        num_leaves=81,
        min_child_samples=29,
        subsample=0.739,
        subsample_freq=1,
        colsample_bytree=0.593,
        reg_lambda=9.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    return Pipeline(
        [("prep", prep), ("model", model)],
    ).set_params(model__categorical_feature=categorical_positions)


def smote_pipeline(numeric: list[str], categorical: list[str], y_train):
    """SMOTE oversampling instead of class weighting.

    The brief names both as valid ways to handle the ~11% positive rate, so this
    settles which one to use with a measurement rather than an assumption. SMOTE
    interpolates synthetic positives in the one-hot feature space, which is
    already a slightly odd thing to do to binary indicator columns — the resampler
    sits inside the CV fold so no synthetic row ever reaches a validation set.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    spec = build_models(y_train)["lightgbm"]
    estimator, _ = spec
    resampled = estimator.__class__(**{**estimator.get_params(), "class_weight": None})
    return ImbPipeline(
        [
            ("prep", assemble(spec, numeric, categorical).named_steps["prep"]),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("model", resampled),
        ]
    )


def soft_vote_pipeline(numeric: list[str], categorical: list[str], y_train) -> VotingClassifier:
    """Average the probabilities of the three tree models.

    They make different errors — random forest bags, the boosters fit residuals
    — so averaging usually buys a little ranking quality for no tuning effort.
    """
    specs = build_models(y_train)
    return VotingClassifier(
        estimators=[
            (name, assemble(specs[name], numeric, categorical))
            for name in ("random_forest", "xgboost", "lightgbm")
        ],
        voting="soft",
        n_jobs=1,
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def cv_score(estimator_factory, X, y, groups) -> dict:
    cv = GroupKFold(n_splits=CV_FOLDS)
    roc, pr = [], []
    for train_idx, val_idx in cv.split(X, y, groups):
        est = estimator_factory()
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = est.predict_proba(X.iloc[val_idx])[:, 1]
        roc.append(roc_auc_score(y.iloc[val_idx], scores))
        pr.append(average_precision_score(y.iloc[val_idx], scores))
    return {
        "roc_auc_mean": float(np.mean(roc)), "roc_auc_std": float(np.std(roc)),
        "pr_auc_mean": float(np.mean(pr)), "pr_auc_std": float(np.std(pr)),
    }


def main() -> pd.DataFrame:
    X, y, groups = build_dataset(include_extra=True)
    X_full, _, y_full, _, g_full, _ = grouped_split(X, y, groups, TEST_SIZE)
    y_arr = y_full.to_numpy()
    print(f"[experiments] {CV_FOLDS}-fold GroupKFold on {len(X_full):,} training rows; "
          f"test set untouched\n")

    def lgbm_on(columns: list[str]):
        """LightGBM restricted to a given feature subset — the ablation workhorse."""
        subset = X_full[columns]
        num, cat = split_feature_types(subset)
        spec = build_models(y_arr)["lightgbm"]
        return subset, (lambda: assemble(spec, num, cat))

    all_cols = list(X_full.columns)
    baseline_cols = [c for c in all_cols if c not in EXTRA_FEATURES]

    variants: dict[str, tuple] = {}

    # A. what src/train.py currently ships
    variants["A_current"] = lgbm_on(baseline_cols)
    # B. drop payer_code — a billing artefact, not a clinical signal
    variants["B_no_payer_code"] = lgbm_on([c for c in baseline_cols if c != "payer_code"])
    # C. add the second wave of engineered features
    variants["C_extra_features"] = lgbm_on(all_cols)
    # D. extra features AND no payer_code
    variants["D_extra_no_payer"] = lgbm_on([c for c in all_cols if c != "payer_code"])

    results = {}
    for name, (subset, factory) in variants.items():
        t0 = time.time()
        results[name] = cv_score(factory, subset, y_full, g_full)
        results[name]["n_features"] = int(subset.shape[1])
        results[name]["seconds"] = round(time.time() - t0, 1)
        r = results[name]
        print(f"[cv] {name:<20} PR-AUC {r['pr_auc_mean']:.4f} ±{r['pr_auc_std']:.4f}  "
              f"ROC-AUC {r['roc_auc_mean']:.4f} ±{r['roc_auc_std']:.4f}  "
              f"({r['n_features']} feats, {r['seconds']}s)")

    # E/F build on whichever feature set won above.
    best_features = max(results, key=lambda k: results[k]["pr_auc_mean"])
    winning_cols = {
        "A_current": baseline_cols,
        "B_no_payer_code": [c for c in baseline_cols if c != "payer_code"],
        "C_extra_features": all_cols,
        "D_extra_no_payer": [c for c in all_cols if c != "payer_code"],
    }[best_features]
    print(f"\n[experiments] best feature set so far: {best_features}\n")

    X_win = X_full[winning_cols]
    num_win, cat_win = split_feature_types(X_win)

    # E. native categorical splits instead of one-hot
    t0 = time.time()
    results["E_native_categorical"] = cv_score(
        lambda: native_categorical_pipeline(num_win, cat_win, y_arr), X_win, y_full, g_full
    )
    results["E_native_categorical"].update(n_features=int(X_win.shape[1]), seconds=round(time.time() - t0, 1))

    # F. soft-voting ensemble of RF + XGB + LGBM
    t0 = time.time()
    results["F_soft_vote_ensemble"] = cv_score(
        lambda: soft_vote_pipeline(num_win, cat_win, y_arr), X_win, y_full, g_full
    )
    results["F_soft_vote_ensemble"].update(n_features=int(X_win.shape[1]), seconds=round(time.time() - t0, 1))

    # G. SMOTE oversampling in place of class weighting
    t0 = time.time()
    results["G_smote"] = cv_score(
        lambda: smote_pipeline(num_win, cat_win, y_arr), X_win, y_full, g_full
    )
    results["G_smote"].update(n_features=int(X_win.shape[1]), seconds=round(time.time() - t0, 1))

    for name in ("E_native_categorical", "F_soft_vote_ensemble", "G_smote"):
        r = results[name]
        print(f"[cv] {name:<20} PR-AUC {r['pr_auc_mean']:.4f} ±{r['pr_auc_std']:.4f}  "
              f"ROC-AUC {r['roc_auc_mean']:.4f} ±{r['roc_auc_std']:.4f}  ({r['seconds']}s)")

    table = pd.DataFrame(results).T.reset_index(names="variant")
    base = results["A_current"]
    table["pr_auc_delta"] = table["pr_auc_mean"] - base["pr_auc_mean"]
    table["roc_auc_delta"] = table["roc_auc_mean"] - base["roc_auc_mean"]
    # Only adopt a change that helps PR-AUC by more than fold noise and does not
    # cost ROC-AUC. "Not worse in any sense" is the bar.
    table["adopt"] = (table["pr_auc_delta"] > base["pr_auc_std"] / 2) & (table["roc_auc_delta"] >= -0.001)

    table.to_csv(REPORT_DIR / "ablation_study.csv", index=False)
    (REPORT_DIR / "ablation_study.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print(table[["variant", "pr_auc_mean", "pr_auc_delta", "roc_auc_mean", "roc_auc_delta", "adopt"]]
          .round(5).to_string(index=False))
    print("=" * 78)
    winner = table.loc[table["pr_auc_mean"].idxmax(), "variant"]
    print(f"\n[experiments] highest CV PR-AUC: {winner}")
    return table


if __name__ == "__main__":
    main()
