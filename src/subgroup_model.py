"""Can the oldest patients be predicted better? Diagnose first, then try fixes.

    python -m src.subgroup_model

The global model ranks the oldest band far worse than the rest (ROC-AUC 0.61 vs
0.78 for the under-40s). The threshold experiment in `src/equity.py` showed that
no thresholding policy repairs this, because thresholds move capacity and cannot
change ranking. So the question is whether the *ranking* can be improved.

Three steps:

1. **Diagnosis.** Univariate AUC of every feature within each age band. If the
   individual features genuinely carry less signal for the oldest patients, the
   problem is the data, and no amount of modelling fixes it. If the features
   still separate but the global model is not exploiting them, the problem is
   the model, and a specialist might.

2. **A dedicated model** fitted on the oldest band alone. It can learn different
   weights, at the cost of far less training data.

3. **Sample weighting**, keeping one global model but upweighting the oldest
   patients so the loss cares more about ranking them correctly.

Every choice is made on validation; the test slice is scored once at the end.
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import (
    AGE_BINS,
    AGE_LABELS,
    FIGURE_DIR,
    OLDEST_BAND,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from .data_prep import build_dataset, split_feature_types
from .pipeline import assemble, build_models
from .train import grouped_split

warnings.filterwarnings("ignore")

OLDEST = OLDEST_BAND


def age_band(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)


# --------------------------------------------------------------------------- #
# 1. Diagnosis
# --------------------------------------------------------------------------- #
def univariate_auc(X: pd.DataFrame, y: pd.Series, bands: pd.Series, numeric: list[str]) -> pd.DataFrame:
    """How much does each single numeric feature separate outcomes, per band?

    Univariate AUC is model-free. If a feature's AUC collapses toward 0.5 inside
    the oldest band, that feature has stopped carrying information for those
    patients — which is a fact about the data, not about the model.
    """
    rows = []
    for feature in numeric:
        row = {"feature": feature}
        for band in AGE_LABELS:
            mask = (bands == band).to_numpy()
            yb = y[mask]
            if mask.sum() < 300 or yb.nunique() < 2:
                row[band] = np.nan
                continue
            values = X.loc[mask, feature]
            if values.nunique() < 2:
                row[band] = np.nan
                continue
            row[band] = roc_auc_score(yb, values)
        rows.append(row)
    out = pd.DataFrame(rows)
    # Distance from 0.5 is the signal; direction does not matter here.
    for band in AGE_LABELS:
        out[f"{band}_signal"] = (out[band] - 0.5).abs()
    return out


# --------------------------------------------------------------------------- #
# 2 & 3. Candidate fixes
# --------------------------------------------------------------------------- #
def fit_global(X_tr, y_tr, numeric, categorical):
    spec = build_models(y_tr.to_numpy())["lightgbm"]
    return assemble(spec, numeric, categorical).fit(X_tr, y_tr)


def fit_specialist(X_tr, y_tr, bands_tr, numeric, categorical):
    """One model fitted on the oldest band alone."""
    mask = (bands_tr == OLDEST).to_numpy()
    Xs, ys = X_tr[mask], y_tr[mask]
    spec = build_models(ys.to_numpy())["lightgbm"]
    print(f"  specialist trains on {len(Xs):,} rows ({ys.mean():.2%} positive)")
    return assemble(spec, numeric, categorical).fit(Xs, ys)


def fit_weighted(X_tr, y_tr, bands_tr, numeric, categorical, weight=3.0):
    """Global model, but the oldest patients count more in the loss."""
    spec = build_models(y_tr.to_numpy())["lightgbm"]
    pipe = assemble(spec, numeric, categorical)
    sw = np.where((bands_tr == OLDEST).to_numpy(), weight, 1.0)
    pipe.fit(X_tr, y_tr, model__sample_weight=sw)
    return pipe


def score_band(pipe, X, y, bands, band=OLDEST) -> dict:
    mask = (bands == band).to_numpy()
    s = pipe.predict_proba(X[mask])[:, 1]
    return {
        "n": int(mask.sum()),
        "roc_auc": float(roc_auc_score(y[mask], s)),
        "pr_auc": float(average_precision_score(y[mask], s)),
    }


def main() -> dict:
    X, y, groups = build_dataset()
    numeric, categorical = split_feature_types(X)

    X_full, X_test, y_full, y_test, g_full, _ = grouped_split(X, y, groups, TEST_SIZE)
    X_tr, X_val, y_tr, y_val, _, _ = grouped_split(X_full, y_full, g_full, VAL_SIZE)

    b_tr, b_val, b_test = (age_band(d["age_mid"]) for d in (X_tr, X_val, X_test))

    print("[bands] encounters per band (train / val / test)")
    for band in AGE_LABELS:
        print(f"  {band:<6} {(b_tr == band).sum():>6,} / {(b_val == band).sum():>6,} "
              f"/ {(b_test == band).sum():>6,}")

    # ---------------- 1. diagnosis ----------------------------------------- #
    print("\n=== diagnosis: does each feature still separate outcomes for 80+? ===")
    uni = univariate_auc(X_full, y_full, age_band(X_full["age_mid"]), numeric)
    uni = uni.sort_values("<40_signal", ascending=False)
    show = uni[["feature"] + AGE_LABELS].head(12)
    print(show.round(3).to_string(index=False))

    mean_signal = {b: float(uni[f"{b}_signal"].mean()) for b in AGE_LABELS}
    print("\n  mean |AUC − 0.5| across features, per band:")
    for b, v in mean_signal.items():
        print(f"    {b:<6} {v:.4f}")
    uni.to_csv(REPORT_DIR / "univariate_auc_by_age.csv", index=False)

    # ---------------- 2 & 3. candidates ------------------------------------ #
    print("\n=== candidate fixes (selection on validation) ===")
    candidates = {}

    print("[global]")
    g_pipe = fit_global(X_tr, y_tr, numeric, categorical)
    candidates["global"] = g_pipe

    print("[specialist]")
    candidates["specialist_80plus"] = fit_specialist(X_tr, y_tr, b_tr, numeric, categorical)

    for w in (3.0, 6.0):
        print(f"[weighted x{w:g}]")
        candidates[f"weighted_x{w:g}"] = fit_weighted(X_tr, y_tr, b_tr, numeric, categorical, w)

    rows = []
    for name, pipe in candidates.items():
        v = score_band(pipe, X_val, y_val, b_val)
        t = score_band(pipe, X_test, y_test, b_test)
        rows.append({
            "model": name,
            "val_roc_auc_80plus": v["roc_auc"], "val_pr_auc_80plus": v["pr_auc"],
            "test_roc_auc_80plus": t["roc_auc"], "test_pr_auc_80plus": t["pr_auc"],
            "n_test_80plus": t["n"],
        })
    table = pd.DataFrame(rows)

    # Overall cost of each candidate, so a 80+ gain that wrecks everyone else is visible.
    for i, (name, pipe) in enumerate(candidates.items()):
        if name == "specialist_80plus":
            table.loc[i, "test_roc_auc_overall"] = np.nan  # only defined on its own band
            continue
        s = pipe.predict_proba(X_test)[:, 1]
        table.loc[i, "test_roc_auc_overall"] = roc_auc_score(y_test, s)
        table.loc[i, "test_pr_auc_overall"] = average_precision_score(y_test, s)

    print("\n" + table.round(4).to_string(index=False))

    baseline = table[table["model"] == "global"].iloc[0]
    best_idx = table["val_roc_auc_80plus"].idxmax()
    best = table.loc[best_idx]
    gain_val = best["val_roc_auc_80plus"] - baseline["val_roc_auc_80plus"]
    gain_test = best["test_roc_auc_80plus"] - baseline["test_roc_auc_80plus"]

    print(f"\n[selection] best on validation: {best['model']} "
          f"(val 80+ ROC-AUC {best['val_roc_auc_80plus']:.4f} vs {baseline['val_roc_auc_80plus']:.4f}, "
          f"{gain_val:+.4f})")
    print(f"[test]      that model scores {best['test_roc_auc_80plus']:.4f} on the 80+ test slice "
          f"vs {baseline['test_roc_auc_80plus']:.4f} for the global model ({gain_test:+.4f})")

    # ---------------- figure ------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    xs = np.arange(len(AGE_LABELS))
    ax.bar(xs, [mean_signal[b] for b in AGE_LABELS], color="#12556B")
    ax.set_xticks(xs)
    ax.set_xticklabels(AGE_LABELS)
    ax.set(xlabel="age band", ylabel="mean |univariate AUC − 0.5|",
           title="Per-feature signal, model-free")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    order = table.sort_values("test_roc_auc_80plus")
    ax.barh(order["model"], order["test_roc_auc_80plus"] - 0.5, left=0.5, color="#dd8452")
    ax.axvline(baseline["test_roc_auc_80plus"], ls="--", c="k", lw=1.0,
               label=f"global = {baseline['test_roc_auc_80plus']:.3f}")
    ax.axvline(0.5, ls=":", c="gray", lw=0.9, label="chance")
    ax.set_xlim(0.5, max(table["test_roc_auc_80plus"].max(), 0.63) + 0.02)
    ax.set(xlabel="ROC-AUC on the 80+ test slice", title="Do any of the fixes help?")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "subgroup_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    table.to_csv(REPORT_DIR / "subgroup_model.csv", index=False)
    summary = {
        "age_bands": {"bins": AGE_BINS, "labels": AGE_LABELS,
                      "note": "age arrives in 10-year brackets, so the top band holds [80-90) and [90-100)"},
        "mean_univariate_signal": mean_signal,
        "candidates": table.to_dict(orient="records"),
        "selected_on_validation": best["model"],
        "gain_val_roc_auc": float(gain_val),
        "gain_test_roc_auc": float(gain_test),
    }
    (REPORT_DIR / "subgroup_model.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n[report] reports/subgroup_model.json · subgroup_model.csv · figures/subgroup_model.png")
    return summary


if __name__ == "__main__":
    main()
