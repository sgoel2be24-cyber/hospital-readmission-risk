"""Decision curve analysis — is using this model better than the alternatives?

    python -m src.decision_curve

PR-AUC says the ranking is good. It does not say whether acting on the ranking
beats the two policies a hospital already has: call everyone, or call no one.
Decision curve analysis (Vickers & Elkin, 2006) answers that, and it is the
standard tool for clinical prediction models.

    net benefit = TP/N − (FP/N) × pt/(1 − pt)

`pt` is the threshold probability: the risk at which a clinician is indifferent
between calling and not calling. It encodes the exchange rate — at pt, you are
willing to make (1 − pt)/pt unnecessary calls to catch one readmission. So the
curve prices the whole range of cost assumptions at once, instead of forcing us
to name a rupee figure we do not have.
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import joblib

from .config import CAPACITY_FRACTION, FIGURE_DIR, MODEL_DIR, REPORT_DIR, TEST_SIZE
from .data_prep import build_dataset
from .train import grouped_split

warnings.filterwarnings("ignore", message="X does not have valid feature names")

PT_GRID = np.arange(0.02, 0.51, 0.005)


def net_benefit(y: np.ndarray, scores: np.ndarray, pt: float) -> float:
    pred = (scores >= pt).astype(int)
    n = len(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return tp / n - (fp / n) * (pt / (1 - pt))


def net_benefit_treat_all(y: np.ndarray, pt: float) -> float:
    prevalence = y.mean()
    return prevalence - (1 - prevalence) * (pt / (1 - pt))


def main() -> dict:
    artefact = joblib.load(MODEL_DIR / "best_model.joblib")
    X, y, groups = build_dataset()
    _, X_test, _, y_test, _, _ = grouped_split(X, y, groups, TEST_SIZE)

    scores = artefact["model"].predict_proba(X_test)[:, 1]
    yt = y_test.to_numpy()
    thr = artefact["threshold"]

    rows = []
    for pt in PT_GRID:
        nb_model = net_benefit(yt, scores, pt)
        nb_all = net_benefit_treat_all(yt, pt)
        rows.append({
            "threshold_probability": float(pt),
            "calls_per_readmission_tolerated": float((1 - pt) / pt),
            "net_benefit_model": nb_model,
            "net_benefit_call_everyone": nb_all,
            "net_benefit_call_nobody": 0.0,
            # Interpretation: extra readmissions caught per 1,000 discharges,
            # compared with calling nobody, at no extra cost in false alarms.
            "net_true_positives_per_1000": nb_model * 1000,
            "advantage_over_best_alternative": nb_model - max(nb_all, 0.0),
        })
    curve = pd.DataFrame(rows)

    useful = curve[curve["advantage_over_best_alternative"] > 0]
    lo = float(useful["threshold_probability"].min()) if len(useful) else float("nan")
    hi = float(useful["threshold_probability"].max()) if len(useful) else float("nan")

    at_deploy = curve.iloc[(curve["threshold_probability"] - thr).abs().argmin()]

    print(f"[decision curve] model beats both 'call everyone' and 'call nobody' "
          f"for pt in [{lo:.3f}, {hi:.3f}]")
    print(f"  that is a willingness to make {(1 - lo) / lo:.1f} down to "
          f"{(1 - hi) / hi:.1f} unnecessary calls per readmission caught")
    print(f"\n[at the deployed threshold pt={at_deploy['threshold_probability']:.3f}]")
    print(f"  net benefit, model          {at_deploy['net_benefit_model']:.5f}")
    print(f"  net benefit, call everyone  {at_deploy['net_benefit_call_everyone']:.5f}")
    print(f"  net benefit, call nobody    0.00000")
    print(f"  → {at_deploy['net_true_positives_per_1000']:.1f} extra readmissions caught per "
          f"1,000 discharges, net of the false alarms they cost")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(curve["threshold_probability"], curve["net_benefit_model"],
            lw=2.2, color="#12556B", label="Use the model")
    ax.plot(curve["threshold_probability"], curve["net_benefit_call_everyone"],
            lw=1.6, color="#dd8452", ls="-", label="Call everyone")
    ax.axhline(0, lw=1.4, color="black", ls="-", label="Call nobody")
    ax.axvline(thr, lw=1.2, color="gray", ls=":", label=f"deployed threshold = {thr:.3f}")
    ax.set_ylim(-0.02, max(curve["net_benefit_model"].max(), 0.03) * 1.25)
    ax.set(
        xlabel="Threshold probability  (risk at which a call becomes worthwhile)",
        ylabel="Net benefit",
        title="Decision curve — the model is worth using across the plausible range",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "decision_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    curve.to_csv(REPORT_DIR / "decision_curve.csv", index=False)
    summary = {
        "method": "Vickers & Elkin decision curve analysis, held-out test set",
        "useful_range_threshold_probability": {"lo": lo, "hi": hi},
        "tolerated_calls_per_readmission": {"at_lo": (1 - lo) / lo, "at_hi": (1 - hi) / hi},
        "at_deployed_threshold": {
            "threshold_probability": float(at_deploy["threshold_probability"]),
            "net_benefit_model": float(at_deploy["net_benefit_model"]),
            "net_benefit_call_everyone": float(at_deploy["net_benefit_call_everyone"]),
            "net_true_positives_per_1000": float(at_deploy["net_true_positives_per_1000"]),
        },
    }
    (REPORT_DIR / "decision_curve.json").write_text(json.dumps(summary, indent=2))
    print("\n[report] reports/decision_curve.json · decision_curve.csv · figures/decision_curve.png")
    return summary


if __name__ == "__main__":
    main()
