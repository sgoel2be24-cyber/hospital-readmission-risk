"""Guards on the parts of the pipeline that fail silently if they break.

    .venv/bin/pytest tests -q

These are correctness checks, not model-quality checks: a data leak or a
mislabelled target still produces a beautiful-looking AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEAD_OR_HOSPICE_DISPOSITIONS, TEST_SIZE, VAL_SIZE
from src.data_prep import (
    EXTRA_FEATURES,
    _age_midpoint,
    _icd9_group,
    build_dataset,
    clean,
    load_raw,
)
from src.train import grouped_split


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


def test_icd9_grouping():
    assert _icd9_group("250.83") == "Diabetes"
    assert _icd9_group("428") == "Circulatory"       # heart failure
    assert _icd9_group("486") == "Respiratory"       # pneumonia
    assert _icd9_group("V57") == "Other"             # supplementary code
    assert _icd9_group("E885") == "Injury"           # external cause
    assert _icd9_group("?") == "Unknown"
    assert _icd9_group(np.nan) == "Unknown"


def test_age_midpoint_is_ordered():
    buckets = ["[0-10)", "[40-50)", "[90-100)"]
    values = [_age_midpoint(b) for b in buckets]
    assert values == [5.0, 45.0, 95.0]
    assert values == sorted(values)


def test_death_and_hospice_encounters_are_removed():
    cleaned = clean(load_raw())
    remaining = set(cleaned["discharge_disposition_id"]) & DEAD_OR_HOSPICE_DISPOSITIONS
    assert not remaining, f"unscoreable dispositions survived cleaning: {remaining}"


def test_identifiers_never_reach_the_feature_matrix(dataset):
    X, _, _ = dataset
    for leaky in ("encounter_id", "patient_nbr", "readmitted", "target"):
        assert leaky not in X.columns


def test_target_matches_the_raw_label(dataset):
    _, y, _ = dataset
    raw = clean(load_raw())
    expected = (raw["readmitted"] == "<30").sum()
    assert y.sum() == expected
    assert 0.10 < y.mean() < 0.13  # documented ~11% positive rate


def test_no_missing_values_survive(dataset):
    X, _, _ = dataset
    assert X.isna().sum().sum() == 0


def test_split_never_puts_a_patient_on_both_sides(dataset):
    """The single most important guard in this project.

    Some patients have many encounters. A row-level split lets the model see the
    same patient in train and test, which inflates every metric.
    """
    X, y, groups = dataset
    X_full, X_test, y_full, y_test, g_full, g_test = grouped_split(X, y, groups, TEST_SIZE)
    _, _, _, _, g_train, g_val = grouped_split(X_full, y_full, g_full, VAL_SIZE)

    assert not set(g_train) & set(g_test)
    assert not set(g_val) & set(g_test)
    assert not set(g_train) & set(g_val)
    assert len(X_full) + len(X_test) == len(X)
    # Class balance should survive the split even though we split on patients.
    assert abs(y_full.mean() - y_test.mean()) < 0.02


def test_split_is_deterministic(dataset):
    """train.py, explain.py and the tests all re-derive the same test set."""
    X, y, groups = dataset
    first = grouped_split(X, y, groups, TEST_SIZE)[1].index
    second = grouped_split(X, y, groups, TEST_SIZE)[1].index
    assert first.equals(second)


def test_prior_encounter_counter_starts_at_zero(dataset):
    X, _, _ = dataset
    assert X["prior_encounters"].min() == 0
    assert (X["prior_encounters"] >= 0).all()


def test_ablated_features_stay_out_of_the_default_model(dataset):
    """The ablation study rejected EXTRA_FEATURES; training must not pick them up.

    They are still computed, so without this guard a later edit could quietly
    reintroduce the slightly-worse 49-feature model.
    """
    X, _, _ = dataset
    assert not set(EXTRA_FEATURES) & set(X.columns)

    X_extra, _, _ = build_dataset(include_extra=True)
    assert set(EXTRA_FEATURES) <= set(X_extra.columns)


def test_engineered_counts_are_consistent(dataset):
    X, _, _ = dataset
    assert (X["n_med_changes"] == X["n_meds_increased"] + X["n_meds_decreased"]).all()
    assert (X["n_med_changes"] <= X["n_meds_active"]).all()
    assert (X["prior_visits_total"] >= X["number_inpatient"]).all()


def test_scoring_a_raw_row_end_to_end():
    """The deployed path must accept a raw EHR row, not a pre-processed one."""
    joblib = pytest.importorskip("joblib")
    from src.config import MODEL_DIR
    from src.predict import score

    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("run `python -m src.train` first")

    artefact = joblib.load(model_path)
    raw = load_raw().head(50)
    scored = score(raw, artefact)

    assert len(scored) > 0
    assert scored["readmission_risk_30d"].between(0, 1).all()
    assert set(scored["flag_for_followup"].unique()) <= {0, 1}
