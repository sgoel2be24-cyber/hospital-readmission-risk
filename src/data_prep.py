"""Load, clean, and feature-engineer the UCI Diabetes 130-US hospitals dataset.

The public entry point is `build_dataset()`, which returns the modelling frame
plus the grouping key used for the patient-level train/test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DEAD_OR_HOSPICE_DISPOSITIONS,
    DROP_COLS,
    KEY_MEDICATION_COLS,
    MEDICATION_COLS,
    POSITIVE_LABEL,
    RAW_CSV,
    TARGET_COL,
)

MISSING_TOKEN = "?"
UNKNOWN = "Unknown"

# The second wave of engineered features, grouped so the ablation study can add
# or remove them as a single block rather than one at a time.
#
# VERDICT: these are computed but excluded by default. The ablation in
# src/experiments.py (reports/ablation_study.csv) shows they cost 0.0005 CV
# PR-AUC and 0.0013 ROC-AUC — inside fold noise, but not an improvement, so the
# simpler 39-feature model is kept. The code stays so the result is reproducible.
EXTRA_FEATURES = [
    "n_distinct_diag_groups",
    "diabetes_in_any_diag",
    "circulatory_in_any_diag",
    "inpatient_share_of_history",
    "emergency_share_of_visits",
    "hba1c_measured",
    "hba1c_tested_and_changed",
    "insulin_changed",
    "lab_intensity",
    "long_stay",
]


# --------------------------------------------------------------------------- #
# ICD-9 diagnosis grouping
# --------------------------------------------------------------------------- #
def _icd9_group(code: object) -> str:
    """Collapse a raw ICD-9 code into one of 9 major diagnosis categories.

    diag_1/2/3 carry ~700-800 distinct codes each; one-hot encoding them raw
    creates thousands of near-empty columns. The grouping below follows the
    chapter structure used in the original study of this dataset.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)) or code == MISSING_TOKEN:
        return UNKNOWN

    code = str(code).strip()
    # V-codes (supplementary) and E-codes (external causes of injury).
    if code.startswith("V"):
        return "Other"
    if code.startswith("E"):
        return "Injury"

    try:
        value = float(code)
    except ValueError:
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"
    if (390 <= value <= 459) or int(value) == 785:
        return "Circulatory"
    if (460 <= value <= 519) or int(value) == 786:
        return "Respiratory"
    if (520 <= value <= 579) or int(value) == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if (580 <= value <= 629) or int(value) == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


# --------------------------------------------------------------------------- #
# Administrative-code grouping
# --------------------------------------------------------------------------- #
def _discharge_group(code: int) -> str:
    if code in (1,):
        return "Home"
    if code in (6, 8):
        return "Home_health"
    if code in (3, 4, 5, 22, 23, 24, 30):
        return "Care_facility"
    if code in (2, 9, 10, 27, 28, 29):
        return "Another_hospital"
    if code in (7,):
        return "Left_AMA"
    if code in (18, 25, 26):
        return UNKNOWN
    return "Other"


def _admission_type_group(code: int) -> str:
    return {
        1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 7: "Trauma",
    }.get(code, UNKNOWN)


def _admission_source_group(code: int) -> str:
    if code in (1, 2, 3):
        return "Referral"
    if code == 7:
        return "Emergency_room"
    if code in (4, 5, 6, 10, 18, 22, 25, 26):
        return "Transfer"
    if code in (11, 12, 13, 14, 23, 24):
        return "Birth"
    return UNKNOWN


def _age_midpoint(bucket: str) -> float:
    """'[70-80)' -> 75.0 — keeps the ordering that one-hot encoding would throw away."""
    low, high = bucket.strip("[)").split("-")
    return (int(low) + int(high)) / 2


def _collapse_rare(series: pd.Series, min_frac: float = 0.01, other: str = "Other") -> pd.Series:
    freq = series.value_counts(normalize=True)
    keep = set(freq[freq >= min_frac].index)
    return series.where(series.isin(keep), other)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    # keep_default_na=False matters: `max_glu_serum` / `A1Cresult` use the literal
    # string "None" to mean "test was never ordered", which pandas would otherwise
    # silently turn into NaN and destroy a genuinely informative level.
    return pd.read_csv(RAW_CSV, keep_default_na=False, na_values=[])


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Row-level filtering and missing-value handling."""
    n_start = len(df)

    # Patients who died or were discharged to hospice cannot be readmitted;
    # leaving them in silently mislabels them as negatives.
    df = df[~df["discharge_disposition_id"].isin(DEAD_OR_HOSPICE_DISPOSITIONS)].copy()
    n_after_death = len(df)

    # 3 encounters carry an unusable gender value.
    df = df[df["gender"] != "Unknown/Invalid"].copy()

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Explicit "Unknown" category rather than row deletion — missingness here is
    # administrative (unrecorded payer / specialty), not random.
    for col in ("race", "payer_code", "medical_specialty"):
        df[col] = df[col].replace(MISSING_TOKEN, UNKNOWN)

    # "None" here means the lab was never ordered, which is itself a signal about
    # how closely the stay was monitored. Rename so it cannot be mistaken for null.
    for col in ("max_glu_serum", "A1Cresult"):
        df[col] = df[col].replace("None", "Not_measured")

    if verbose:
        print(
            f"[clean] {n_start:,} -> {n_after_death:,} rows after removing death/hospice discharges "
            f"-> {len(df):,} after gender filter"
        )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the modelling features. All of these are knowable at discharge."""
    out = df.copy()

    # --- Diagnoses -------------------------------------------------------- #
    for col in ("diag_1", "diag_2", "diag_3"):
        out[f"{col}_group"] = out[col].map(_icd9_group)
    out = out.drop(columns=["diag_1", "diag_2", "diag_3"])

    # --- Administrative codes --------------------------------------------- #
    out["discharge_group"] = out["discharge_disposition_id"].map(_discharge_group)
    out["admission_type_group"] = out["admission_type_id"].map(_admission_type_group)
    out["admission_source_group"] = out["admission_source_id"].map(_admission_source_group)
    out = out.drop(columns=["discharge_disposition_id", "admission_type_id", "admission_source_id"])

    # --- Demographics ------------------------------------------------------ #
    out["age_mid"] = out["age"].map(_age_midpoint)
    out = out.drop(columns=["age"])

    # --- Prior utilisation: the strongest documented readmission driver ---- #
    out["prior_visits_total"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )
    out["has_prior_inpatient"] = (out["number_inpatient"] > 0).astype(int)
    out["has_prior_emergency"] = (out["number_emergency"] > 0).astype(int)

    # Number of this patient's *earlier* encounters. Known at discharge time, and
    # patients never straddle the train/test split, so this cannot leak.
    #
    # At serving time a single encounter carries no history of its own, so the
    # caller may supply this count from the EHR; only fall back to deriving it
    # from the batch when it is absent. Without that escape hatch, live scoring
    # would silently send every patient in as a first-time admission.
    if "prior_encounters" not in out.columns:
        out = out.sort_values("encounter_id")
        out["prior_encounters"] = out.groupby("patient_nbr").cumcount()

    # --- Medication burden and regimen churn ------------------------------- #
    med_cols = [c for c in MEDICATION_COLS if c in out.columns]
    med_frame = out[med_cols]
    out["n_meds_active"] = (med_frame != "No").sum(axis=1)
    out["n_med_changes"] = med_frame.isin(["Up", "Down"]).sum(axis=1)
    out["n_meds_increased"] = (med_frame == "Up").sum(axis=1)
    out["n_meds_decreased"] = (med_frame == "Down").sum(axis=1)

    # Keep only the widely-prescribed medications as individual categoricals;
    # the long tail is captured by the counts above.
    out = out.drop(columns=[c for c in med_cols if c not in KEY_MEDICATION_COLS])

    # --- Intensity of the stay -------------------------------------------- #
    days = out["time_in_hospital"].clip(lower=1)
    out["meds_per_day"] = out["num_medications"] / days
    out["procedures_per_day"] = (out["num_procedures"] + out["num_lab_procedures"]) / days

    # --- Second wave: comorbidity spread and care intensity ---------------- #
    # Listed in EXTRA_FEATURES so the ablation in src/experiments.py can drop
    # them as a block and prove they earn their place.
    diag_groups = out[["diag_1_group", "diag_2_group", "diag_3_group"]]
    out["n_distinct_diag_groups"] = diag_groups.nunique(axis=1)
    out["diabetes_in_any_diag"] = (diag_groups == "Diabetes").any(axis=1).astype(int)
    out["circulatory_in_any_diag"] = (diag_groups == "Circulatory").any(axis=1).astype(int)

    # How concentrated this patient's history is: 4 admissions in 5 recorded
    # encounters is a different patient from 4 in 20.
    out["inpatient_share_of_history"] = out["number_inpatient"] / (out["prior_encounters"] + 1)
    out["emergency_share_of_visits"] = out["number_emergency"] / out["prior_visits_total"].clip(lower=1)

    # The original study of this dataset found HbA1c testing mattered mainly in
    # combination with what was done about it — a test with no regimen change is
    # a different signal from a test followed by one.
    out["hba1c_measured"] = (out["A1Cresult"] != "Not_measured").astype(int)
    out["hba1c_tested_and_changed"] = (
        out["hba1c_measured"] & (out["change"] == "Ch")
    ).astype(int)
    out["insulin_changed"] = out["insulin"].isin(["Up", "Down"]).astype(int)

    out["lab_intensity"] = out["num_lab_procedures"] / out["number_diagnoses"].clip(lower=1)
    out["long_stay"] = (out["time_in_hospital"] >= 7).astype(int)

    # --- Rare-level collapsing -------------------------------------------- #
    out["medical_specialty"] = _collapse_rare(out["medical_specialty"], min_frac=0.01)
    out["payer_code"] = _collapse_rare(out["payer_code"], min_frac=0.01)

    # --- Target ------------------------------------------------------------ #
    # A live encounter being scored at discharge has no outcome yet, so the label
    # is only built when it is actually present.
    if TARGET_COL in out.columns:
        out["target"] = (out[TARGET_COL] == POSITIVE_LABEL).astype(int)
        out = out.drop(columns=[TARGET_COL])

    return out


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for the feature matrix."""
    features = df.drop(columns=["target", "encounter_id", "patient_nbr"], errors="ignore")
    categorical = [c for c in features.columns if features[c].dtype == object]
    numeric = [c for c in features.columns if c not in categorical]
    return numeric, categorical


def build_dataset(include_extra: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, groups) ready for a patient-level split.

    `groups` is `patient_nbr`, kept out of X so it can never leak into a model.

    `include_extra=True` adds EXTRA_FEATURES; the ablation study uses it, and
    training does not, because they did not improve cross-validated performance.
    """
    df = engineer_features(clean(load_raw()))
    y = df["target"]
    groups = df["patient_nbr"]
    X = df.drop(columns=["target", "encounter_id", "patient_nbr"])
    if not include_extra:
        X = X.drop(columns=[c for c in EXTRA_FEATURES if c in X.columns])
    print(
        f"[build] X={X.shape}  positives={y.sum():,} ({y.mean():.2%})  "
        f"unique patients={groups.nunique():,}"
    )
    return X, y, groups
