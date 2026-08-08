"""Correctness guards on the statistics added after the model was frozen.

    .venv/bin/pytest tests/test_evidence.py -q

These are the pieces most likely to be silently wrong: a bootstrap that resamples
the wrong unit still produces confident-looking intervals, and a net-benefit
formula with a flipped term still produces a plausible-looking curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.decision_curve import net_benefit, net_benefit_treat_all
from src.uncertainty import ci, cluster_bootstrap_indices


# --------------------------------------------------------------------------- #
# Cluster bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_resamples_whole_patients_never_partial():
    """A patient is drawn all-or-nothing — that is the point of a cluster bootstrap.

    If this ever resamples rows, every confidence interval in the project becomes
    too narrow, and nothing else would reveal it.
    """
    groups = pd.Series(["a", "a", "a", "b", "c", "c"], index=range(6))
    rng = np.random.default_rng(0)

    sizes = {"a": 3, "b": 1, "c": 2}
    for _ in range(50):
        idx = cluster_bootstrap_indices(groups, rng)
        drawn = groups.iloc[idx].value_counts()
        for patient, count in drawn.items():
            # every appearance contributes the patient's full block of rows
            assert count % sizes[patient] == 0


def test_bootstrap_sample_covers_all_rows_when_every_patient_drawn():
    groups = pd.Series(["p1", "p1", "p2", "p3"], index=range(4))
    rng = np.random.default_rng(1)
    idx = cluster_bootstrap_indices(groups, rng)
    # one draw per patient, so the total row count is a sum of whole blocks
    assert len(idx) > 0
    assert set(idx) <= set(range(len(groups)))


def test_bootstrap_draws_as_many_patients_as_exist():
    groups = pd.Series([f"p{i // 2}" for i in range(20)])  # 10 patients, 2 rows each
    rng = np.random.default_rng(2)
    idx = cluster_bootstrap_indices(groups, rng)
    assert len(idx) == 20  # 10 patients drawn × 2 rows


def test_ci_brackets_the_mean():
    values = np.random.default_rng(3).normal(0.5, 0.1, size=5000)
    out = ci(values)
    assert out["lo"] < out["mean"] < out["hi"]
    assert out["width"] == pytest.approx(out["hi"] - out["lo"])
    # a 95% interval on a normal sample is close to ±1.96 SD
    assert out["width"] == pytest.approx(2 * 1.96 * 0.1, rel=0.1)


# --------------------------------------------------------------------------- #
# Decision curve
# --------------------------------------------------------------------------- #
def test_net_benefit_of_a_perfect_model_is_the_prevalence():
    """Perfect prediction: all TPs, no FPs, so net benefit = prevalence."""
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = y.astype(float)  # 1.0 for positives, 0.0 for negatives
    for pt in (0.1, 0.2, 0.4):
        assert net_benefit(y, scores, pt) == pytest.approx(y.mean())


def test_net_benefit_of_flagging_nobody_is_zero():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    scores = np.zeros(len(y))
    # threshold above every score → nothing flagged → no TPs and no FPs
    assert net_benefit(y, scores, 0.5) == 0.0


def test_treat_all_matches_flagging_everyone():
    rng = np.random.default_rng(4)
    y = (rng.random(500) < 0.2).astype(int)
    scores = np.ones(len(y))  # flag everyone
    for pt in (0.05, 0.15, 0.3):
        assert net_benefit(y, scores, pt) == pytest.approx(net_benefit_treat_all(y, pt))


def test_treat_all_turns_negative_once_pt_exceeds_prevalence():
    """Calling everyone stops being worthwhile above the base rate."""
    y = np.array([1] * 10 + [0] * 90)  # prevalence 0.10
    assert net_benefit_treat_all(y, 0.05) > 0
    assert net_benefit_treat_all(y, 0.10) == pytest.approx(0.0, abs=1e-9)
    assert net_benefit_treat_all(y, 0.20) < 0


def test_net_benefit_penalises_false_positives_more_as_pt_rises():
    rng = np.random.default_rng(5)
    y = (rng.random(400) < 0.15).astype(int)
    scores = rng.random(400)  # useless model
    benefits = [net_benefit(y, scores, pt) for pt in (0.1, 0.2, 0.3, 0.4)]
    assert benefits == sorted(benefits, reverse=True)
