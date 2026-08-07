"""Batch scoring against the saved model — the offline half of deployment.

    python -m src.predict --input data/diabetic_data.csv --output reports/scored_discharges.csv
    python -m src.predict --demo

Takes raw encounter rows in the same shape as the source CSV, applies the exact
preprocessing used in training, and returns a risk score plus a follow-up flag.
"""

from __future__ import annotations

import argparse
import warnings

import joblib
import pandas as pd

from .config import MEDICATION_COLS, MODEL_DIR, REPORT_DIR
from .data_prep import clean, engineer_features, load_raw

# Cosmetic LightGBM warning; see the note in train.py.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Columns `clean`/`engineer_features` cannot run without. `readmitted` is
# deliberately absent: at discharge the outcome does not exist yet.
REQUIRED_RAW_COLUMNS = [
    "discharge_disposition_id", "admission_type_id", "admission_source_id",
    "gender", "age", "race", "diag_1", "diag_2", "diag_3",
    "time_in_hospital", "num_medications", "num_procedures", "num_lab_procedures",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed",
]

_ARTEFACT = None


def load_artefact(path=None) -> dict:
    """Load the trained model once and reuse it — loading costs ~100ms."""
    global _ARTEFACT
    if _ARTEFACT is None:
        _ARTEFACT = joblib.load(path or MODEL_DIR / "best_model.joblib")
    return _ARTEFACT


def prepare(raw: pd.DataFrame, artefact: dict) -> pd.DataFrame:
    """Run raw encounter rows through the training-time feature pipeline.

    Deliberately reuses `clean` and `engineer_features` rather than duplicating
    the logic — training/serving skew in the feature code is the classic way a
    model that scored well offline degrades quietly in production.
    """
    absent = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if absent:
        raise ValueError(f"input is missing required raw columns: {absent}")

    # Medication columns default to "No" when a hospital's export omits them.
    filled = raw.copy()
    for col in MEDICATION_COLS:
        if col not in filled.columns:
            filled[col] = "No"

    prepared = engineer_features(clean(filled, verbose=False))
    missing = [c for c in artefact["feature_columns"] if c not in prepared.columns]
    if missing:
        raise ValueError(f"input is missing required columns after preparation: {missing}")
    return prepared[artefact["feature_columns"]]


def score(raw: pd.DataFrame, artefact: dict | None = None) -> pd.DataFrame:
    """Return one row per encounter: risk, flag, and risk band."""
    artefact = artefact or load_artefact()
    features = prepare(raw, artefact)
    risk = artefact["model"].predict_proba(features)[:, 1]

    out = pd.DataFrame(index=features.index)
    out["readmission_risk_30d"] = risk
    out["flag_for_followup"] = (risk >= artefact["threshold"]).astype(int)
    # Bands give the ward something coarser than a decimal to act on.
    out["risk_band"] = pd.cut(
        risk,
        bins=[-0.01, 0.08, artefact["threshold"], 0.30, 1.01],
        labels=["Low", "Moderate", "High", "Very high"],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score discharges for 30-day readmission risk")
    parser.add_argument("--input", default=None, help="CSV in the raw dataset schema")
    parser.add_argument("--output", default=str(REPORT_DIR / "scored_discharges.csv"))
    parser.add_argument("--demo", action="store_true", help="score 20 rows from the bundled dataset")
    args = parser.parse_args()

    artefact = load_artefact()
    raw = load_raw() if args.input is None else pd.read_csv(args.input, keep_default_na=False, na_values=[])
    if args.demo:
        raw = raw.head(500)

    scored = score(raw, artefact)
    result = pd.concat([raw.loc[scored.index, ["encounter_id", "patient_nbr"]], scored], axis=1)

    if args.demo:
        print(f"model={artefact['model_name']} ({artefact['calibration']}-calibrated)  "
              f"threshold={artefact['threshold']:.4f}")
        print(result.head(20).to_string(index=False))
        print(f"\nflagged {result['flag_for_followup'].mean():.1%} of {len(result)} encounters")
    else:
        result.to_csv(args.output, index=False)
        print(f"wrote {len(result):,} scored encounters -> {args.output}")
        print(result["risk_band"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
