"""Is 0.684 real, and is it the ceiling?

    python -m src.robustness

Two diagnostics that the headline number depends on but never justified:

  seed sweep      The entire result rests on one GroupShuffleSplit with
                  random_state=42. If that split happened to be favourable,
                  nothing else in the project would reveal it. Re-runs the full
                  pipeline across several seeds and reports the spread.

  learning curve  The deck claims we have hit the information ceiling of
                  administrative data. That is currently inferred from seven
                  ablation variants converging. A flat learning curve is direct
                  evidence; a rising one would falsify the claim.

The learning curve is scored on **validation**, not test, so the "test set is
read once" guarantee survives.
"""

from __future__ import annotations

import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .config import (
    CAPACITY_FRACTION,
    FIGURE_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from .data_prep import build_dataset, split_feature_types
from .evaluate import capacity_threshold, metrics_at_threshold
from .pipeline import assemble, build_models
from .train import grouped_split

warnings.filterwarnings("ignore")

SEEDS = [42, 1, 7, 13, 101, 2024, 31337]
FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]


def _fit_lightgbm(X_tr, y_tr, numeric, categorical):
    spec = build_models(y_tr.to_numpy())["lightgbm"]
    pipe = assemble(spec, numeric, categorical)
    pipe.fit(X_tr, y_tr)
    return pipe


# --------------------------------------------------------------------------- #
# Seed sweep
# --------------------------------------------------------------------------- #
def seed_sweep(X, y, groups, numeric, categorical) -> pd.DataFrame:
    """Re-split and re-train end to end under different random seeds."""
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        X_full, X_test, y_full, y_test, g_full, g_test = grouped_split(X, y, groups, TEST_SIZE, seed=seed)
        X_tr, X_val, y_tr, y_val, _, _ = grouped_split(X_full, y_full, g_full, VAL_SIZE, seed=seed)

        pipe = _fit_lightgbm(X_tr, y_tr, numeric, categorical)
        val_scores = pipe.predict_proba(X_val)[:, 1]
        test_scores = pipe.predict_proba(X_test)[:, 1]

        # Threshold still comes from validation, exactly as in training.
        thr = capacity_threshold(val_scores, CAPACITY_FRACTION)
        at_cap = metrics_at_threshold(y_test.to_numpy(), test_scores, thr)

        rows.append({
            "seed": seed,
            "test_roc_auc": roc_auc_score(y_test, test_scores),
            "test_pr_auc": average_precision_score(y_test, test_scores),
            "recall_at_capacity": at_cap["recall"],
            "precision_at_capacity": at_cap["precision"],
            "lift": at_cap["lift_over_base_rate"],
            "test_positive_rate": float(y_test.mean()),
            "seconds": round(time.time() - t0, 1),
        })
        r = rows[-1]
        print(f"[seed {seed:>6}] ROC-AUC {r['test_roc_auc']:.4f}  PR-AUC {r['test_pr_auc']:.4f}  "
              f"recall {r['recall_at_capacity']:.3f}  ({r['seconds']}s)")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Learning curve
# --------------------------------------------------------------------------- #
def learning_curve(X, y, groups, numeric, categorical) -> pd.DataFrame:
    """PR-AUC against training size, subsampled by patient, scored on validation."""
    X_full, _, y_full, _, g_full, _ = grouped_split(X, y, groups, TEST_SIZE)
    X_tr, X_val, y_tr, y_val, g_tr, _ = grouped_split(X_full, y_full, g_full, VAL_SIZE)

    rows = []
    for frac in FRACTIONS:
        if frac >= 1.0:
            idx = np.arange(len(X_tr))
        else:
            # Subsample patients, not rows — a half-sized training set must still
            # contain whole patients or the comparison is not like for like.
            splitter = GroupShuffleSplit(n_splits=1, train_size=frac, random_state=RANDOM_STATE)
            idx, _ = next(splitter.split(X_tr, y_tr, g_tr))

        t0 = time.time()
        pipe = _fit_lightgbm(X_tr.iloc[idx], y_tr.iloc[idx], numeric, categorical)
        scores = pipe.predict_proba(X_val)[:, 1]
        rows.append({
            "train_fraction": frac,
            "n_train_rows": int(len(idx)),
            "n_train_patients": int(g_tr.iloc[idx].nunique()),
            "val_roc_auc": roc_auc_score(y_val, scores),
            "val_pr_auc": average_precision_score(y_val, scores),
            "seconds": round(time.time() - t0, 1),
        })
        r = rows[-1]
        print(f"[curve {frac:>5.0%}] {r['n_train_rows']:>6,} rows  "
              f"val PR-AUC {r['val_pr_auc']:.4f}  ROC-AUC {r['val_roc_auc']:.4f}  ({r['seconds']}s)")

    return pd.DataFrame(rows)


def main() -> dict:
    X, y, groups = build_dataset()
    numeric, categorical = split_feature_types(X)

    print("\n=== seed sweep: does the result survive a different split? ===")
    seeds = seed_sweep(X, y, groups, numeric, categorical)

    print("\n=== learning curve: would more data help? ===")
    curve = learning_curve(X, y, groups, numeric, categorical)

    seeds.to_csv(REPORT_DIR / "seed_sweep.csv", index=False)
    curve.to_csv(REPORT_DIR / "learning_curve.csv", index=False)

    # --- figures ----------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    mean = seeds["test_pr_auc"].mean()
    sd = seeds["test_pr_auc"].std(ddof=1)
    ax.axhspan(mean - sd, mean + sd, color="#dd8452", alpha=0.15, label="±1 SD")
    ax.axhline(mean, color="#dd8452", ls="--", lw=1.2, label=f"mean = {mean:.4f}")
    # Seeds have no natural order, so no connecting line — points only.
    colours = ["#C44E52" if s == RANDOM_STATE else "#12556B" for s in seeds["seed"]]
    ax.scatter(range(len(seeds)), seeds["test_pr_auc"], c=colours, s=55, zorder=3)
    ax.annotate("reported\nsplit", xy=(0, seeds["test_pr_auc"].iloc[0]),
                xytext=(0.45, seeds["test_pr_auc"].iloc[0] - 0.0015),
                fontsize=8.5, color="#C44E52")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds["seed"], rotation=45, fontsize=9)
    ax.set_xlim(-0.5, len(seeds) - 0.5)
    ax.set(xlabel="split seed", ylabel="Test PR-AUC",
           title=f"Seed sensitivity (spread {seeds['test_pr_auc'].max() - seeds['test_pr_auc'].min():.4f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(curve["n_train_rows"], curve["val_pr_auc"], "o-", color="#12556B", lw=1.8, ms=6)
    ax.set(xlabel="training encounters", ylabel="Validation PR-AUC",
           title="Learning curve — does more data help?")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- verdicts ---------------------------------------------------------- #
    spread = float(seeds["test_pr_auc"].max() - seeds["test_pr_auc"].min())
    last_two = curve["val_pr_auc"].iloc[-2:].to_numpy()
    tail_gain = float(last_two[-1] - last_two[0])

    summary = {
        "seed_sweep": {
            "seeds": SEEDS,
            "test_pr_auc_mean": float(seeds["test_pr_auc"].mean()),
            "test_pr_auc_std": float(seeds["test_pr_auc"].std(ddof=1)),
            "test_pr_auc_min": float(seeds["test_pr_auc"].min()),
            "test_pr_auc_max": float(seeds["test_pr_auc"].max()),
            "test_pr_auc_spread": spread,
            "test_roc_auc_mean": float(seeds["test_roc_auc"].mean()),
            "test_roc_auc_std": float(seeds["test_roc_auc"].std(ddof=1)),
            "recall_mean": float(seeds["recall_at_capacity"].mean()),
            "recall_std": float(seeds["recall_at_capacity"].std(ddof=1)),
            "rows": seeds.to_dict(orient="records"),
        },
        "learning_curve": {
            "fractions": FRACTIONS,
            "gain_over_last_15pct_of_data": tail_gain,
            "rows": curve.to_dict(orient="records"),
        },
    }
    (REPORT_DIR / "robustness.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n[seed sweep]  PR-AUC {seeds['test_pr_auc'].mean():.4f} ± {seeds['test_pr_auc'].std(ddof=1):.4f} "
          f"(min {seeds['test_pr_auc'].min():.4f}, max {seeds['test_pr_auc'].max():.4f})")
    print(f"[learning]    last 15% of training data moved validation PR-AUC by {tail_gain:+.4f}")
    print("[report] reports/seed_sweep.csv · reports/learning_curve.csv · figures/robustness.png")
    return summary


if __name__ == "__main__":
    main()
