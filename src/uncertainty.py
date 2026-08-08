"""Confidence intervals for every headline number.

    python -m src.uncertainty

Point estimates on a single test set say nothing about how much of the result is
sampling luck. This runs a **cluster bootstrap** — resampling *patients*, not
rows — over the held-out test set, and reports:

  * a 95% interval for each headline metric of the deployed model;
  * paired intervals for the difference between models, using identical
    resamples, which is the only honest way to answer "is LightGBM actually
    better than XGBoost?"

Resampling rows would break the same assumption a row-level train/test split
breaks: encounters from one patient are correlated, so treating them as
independent draws produces intervals that are too narrow.
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import joblib

from .config import (
    CAPACITY_FRACTION,
    FIGURE_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from .data_prep import build_dataset
from .evaluate import capacity_threshold
from .train import grouped_split

warnings.filterwarnings("ignore", message="X does not have valid feature names")

N_BOOT = 2000
ALPHA = 0.05  # → 95% intervals


def cluster_bootstrap_indices(groups: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Resample patients with replacement, return the row positions they own.

    A patient sampled twice contributes all of their encounters twice, which is
    what preserves the within-patient correlation structure.
    """
    codes, _ = pd.factorize(groups.to_numpy())
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    # start offset of each patient's block in the sorted order
    starts = np.searchsorted(sorted_codes, np.arange(codes.max() + 1))
    ends = np.append(starts[1:], len(codes))

    picked = rng.integers(0, len(starts), size=len(starts))
    return np.concatenate([order[starts[p]:ends[p]] for p in picked])


def ci(values: np.ndarray) -> dict:
    lo, hi = np.quantile(values, [ALPHA / 2, 1 - ALPHA / 2])
    return {
        "mean": float(np.mean(values)),
        "lo": float(lo),
        "hi": float(hi),
        "width": float(hi - lo),
    }


def main() -> dict:
    rng = np.random.default_rng(RANDOM_STATE)

    artefact = joblib.load(MODEL_DIR / "best_model.joblib")
    fitted = joblib.load(MODEL_DIR / "all_models.joblib")

    X, y, groups = build_dataset()
    X_full, X_test, y_full, y_test, g_full, g_test = grouped_split(X, y, groups, TEST_SIZE)
    _, X_val, _, y_val, _, _ = grouped_split(X_full, y_full, g_full, VAL_SIZE)

    yt = y_test.to_numpy()
    thr = artefact["threshold"]

    # Pre-compute every model's test scores once; the bootstrap only indexes.
    scores = {name: pipe.predict_proba(X_test)[:, 1] for name, pipe in fitted.items()}
    scores["deployed (calibrated)"] = artefact["model"].predict_proba(X_test)[:, 1]
    print(f"[bootstrap] {N_BOOT} cluster resamples over {g_test.nunique():,} test patients")

    metric_names = ["roc_auc", "pr_auc", "recall_at_capacity", "precision_at_capacity", "brier"]
    draws = {name: {m: [] for m in metric_names} for name in scores}

    for b in range(N_BOOT):
        idx = cluster_bootstrap_indices(g_test, rng)
        yb = yt[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        for name, s in scores.items():
            sb = s[idx]
            pred = (sb >= thr).astype(int)
            d = draws[name]
            d["roc_auc"].append(roc_auc_score(yb, sb))
            d["pr_auc"].append(average_precision_score(yb, sb))
            d["recall_at_capacity"].append(recall_score(yb, pred, zero_division=0))
            d["precision_at_capacity"].append(precision_score(yb, pred, zero_division=0))
            d["brier"].append(brier_score_loss(yb, sb))
        if (b + 1) % 500 == 0:
            print(f"  {b + 1}/{N_BOOT}")

    summary = {
        name: {m: ci(np.asarray(v)) for m, v in d.items()} for name, d in draws.items()
    }

    print("\n[deployed model — 95% CI on held-out test]")
    dep = summary["deployed (calibrated)"]
    for m in metric_names:
        c = dep[m]
        print(f"  {m:<24} {c['mean']:.4f}  [{c['lo']:.4f}, {c['hi']:.4f}]")

    # --- paired differences ------------------------------------------------ #
    # Same resample for both models on every iteration, so the interval is on
    # the *difference* and cancels the shared sampling noise.
    pairs = [("lightgbm", "xgboost"), ("lightgbm", "random_forest"),
             ("lightgbm", "logistic_regression"), ("xgboost", "random_forest")]
    paired = {}
    print("\n[paired differences — 95% CI on Δ PR-AUC]")
    for a, b_ in pairs:
        if a not in draws or b_ not in draws:
            continue
        diff = np.asarray(draws[a]["pr_auc"]) - np.asarray(draws[b_]["pr_auc"])
        c = ci(diff)
        # An interval straddling zero means the ordering is not resolved.
        c["significant"] = bool(c["lo"] > 0 or c["hi"] < 0)
        paired[f"{a} - {b_}"] = c
        verdict = "resolved" if c["significant"] else "NOT resolved (CI spans 0)"
        print(f"  {a} - {b_:<20} {c['mean']:+.4f}  [{c['lo']:+.4f}, {c['hi']:+.4f}]  {verdict}")

    # --- figure ------------------------------------------------------------ #
    order = [n for n in ["logistic_regression", "random_forest", "xgboost", "lightgbm", "mlp"] if n in summary]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, name in enumerate(order):
        c = summary[name]["pr_auc"]
        ax.plot([c["lo"], c["hi"]], [i, i], color="#4c72b0", lw=2.4, solid_capstyle="round")
        ax.plot(c["mean"], i, "o", color="#12556B", ms=7)
    ax.axvline(float(np.mean(yt)), ls="--", c="k", lw=0.9, label=f"base rate = {yt.mean():.3f}")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Test PR-AUC (95% cluster-bootstrap CI)")
    ax.set_title("Overlapping intervals mean the ordering is not resolved")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bootstrap_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    out = {
        "n_bootstrap": N_BOOT,
        "resampling": "cluster bootstrap on patient_nbr over the held-out test set",
        "n_test_patients": int(g_test.nunique()),
        "n_test_encounters": int(len(X_test)),
        "threshold": float(thr),
        "capacity_fraction": CAPACITY_FRACTION,
        "per_model": summary,
        "paired_differences_pr_auc": paired,
    }
    (REPORT_DIR / "bootstrap_ci.json").write_text(json.dumps(out, indent=2))
    print(f"\n[report] reports/bootstrap_ci.json · figures/bootstrap_ci.png")
    return out


if __name__ == "__main__":
    main()
