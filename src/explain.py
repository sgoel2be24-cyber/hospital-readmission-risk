"""Feature attribution for the deployed model.

    python -m src.explain

Two views, because they answer different questions:
  * permutation importance on the held-out test set — which features the model
    actually relies on to generalise;
  * SHAP — the direction and magnitude of each feature's effect, and the
    per-patient reason codes a nurse would need to see next to a risk score.
"""

from __future__ import annotations

import os
import warnings

# permutation_importance fans out to loky worker processes, which do not inherit
# the parent's warning filters — this is how they get told to stay quiet about
# LightGBM's cosmetic feature-name warning. Must be set before joblib imports.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from .config import CAPACITY_FRACTION, FIGURE_DIR, MODEL_DIR, RANDOM_STATE, REPORT_DIR, TEST_SIZE
from .data_prep import build_dataset, split_feature_types
from .train import grouped_split

warnings.filterwarnings("ignore")

SHAP_SAMPLE = 3000  # rows sampled for SHAP; exact tree SHAP on 20k rows is wasteful
PERM_SAMPLE = 6000


def _unwrap(artefact) -> tuple:
    """Pull the underlying (preprocessor, booster) out of the calibrated wrapper.

    CalibratedClassifierCV stores the frozen pipeline; TreeExplainer needs the raw
    booster and the already-encoded matrix, not the sklearn wrapper.
    """
    calibrated = artefact["model"]
    frozen = calibrated.calibrated_classifiers_[0].estimator
    pipeline = getattr(frozen, "estimator", frozen)  # FrozenEstimator -> Pipeline
    return pipeline.named_steps["prep"], pipeline.named_steps["model"], calibrated


def main() -> pd.DataFrame:
    artefact = joblib.load(MODEL_DIR / "best_model.joblib")
    prep, booster, calibrated = _unwrap(artefact)

    X, y, groups = build_dataset()
    _, X_test, _, y_test, _, _ = grouped_split(X, y, groups, TEST_SIZE)

    feature_names = list(prep.get_feature_names_out())

    # --- Permutation importance (model-agnostic, on held-out data) --------- #
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(PERM_SAMPLE, len(X_test)), replace=False)
    perm = permutation_importance(
        calibrated, X_test.iloc[idx], y_test.iloc[idx],
        scoring="average_precision", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    perm_df.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    print("[permutation] top 12 raw features by PR-AUC drop when shuffled:")
    print(perm_df.head(12).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    top = perm_df.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#55a868")
    ax.set(xlabel="Drop in PR-AUC when shuffled", title="Permutation importance (test set)")
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(FIGURE_DIR / "permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- SHAP -------------------------------------------------------------- #
    sample_idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    X_encoded = prep.transform(X_sample)
    encoded_df = pd.DataFrame(X_encoded, columns=feature_names, index=X_sample.index)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(encoded_df)
    if isinstance(shap_values, list):  # some versions return one array per class
        shap_values = shap_values[1]

    shap_df = (
        pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_df.to_csv(REPORT_DIR / "shap_importance.csv", index=False)
    print("\n[shap] top 15 encoded features by mean |SHAP|:")
    print(shap_df.head(15).to_string(index=False))

    shap.summary_plot(shap_values, encoded_df, max_display=18, show=False)
    plt.title("SHAP — direction and magnitude of each feature's effect", fontsize=11)
    plt.savefig(FIGURE_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, encoded_df, plot_type="bar", max_display=18, show=False)
    plt.title("SHAP — mean absolute impact on the risk score", fontsize=11)
    plt.savefig(FIGURE_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Per-patient reason codes ------------------------------------------ #
    # What the ward actually receives: a score plus the three or four facts that
    # pushed it up. A flagged patient with no explainable driver is not actionable.
    risk = calibrated.predict_proba(X_sample)[:, 1]
    highest = np.argsort(risk)[-5:][::-1]
    reasons = []
    for pos in highest:
        contrib = pd.Series(shap_values[pos], index=feature_names).sort_values(ascending=False)
        reasons.append(
            {
                "row": int(X_sample.index[pos]),
                "predicted_risk": float(risk[pos]),
                "actually_readmitted": int(y_test.loc[X_sample.index[pos]]),
                "top_drivers": "; ".join(
                    f"{name}={encoded_df.iloc[pos][name]:g} (+{val:.3f})"
                    for name, val in contrib.head(4).items()
                ),
            }
        )
    reason_df = pd.DataFrame(reasons)
    reason_df.to_csv(REPORT_DIR / "example_reason_codes.csv", index=False)
    print(f"\n[reason codes] 5 highest-risk test patients (threshold={artefact['threshold']:.3f}, "
          f"flagging top {CAPACITY_FRACTION:.0%}):")
    print(reason_df.to_string(index=False))

    print(f"\n[done] figures in {FIGURE_DIR}")
    return shap_df


if __name__ == "__main__":
    main()
