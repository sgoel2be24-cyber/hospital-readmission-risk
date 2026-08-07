"""The deployment path: raw EHR row in, risk score out.

Training-time metrics say nothing about whether the service works. These tests
exercise the same feature code the model was fitted on, through the API.
"""

from __future__ import annotations

import pytest

from src.config import MODEL_DIR
from src.data_prep import load_raw

pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "best_model.joblib").exists(),
    reason="run `python -m src.train` first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api import app

    return TestClient(app)


@pytest.fixture(scope="module")
def raw_rows():
    return load_raw().head(30)


def test_health_reports_the_deployed_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert 0 < body["threshold"] < 1
    assert body["flags_top_fraction"] == 0.20


def test_score_without_the_outcome_column(raw_rows, client):
    """The label does not exist at discharge — scoring must not depend on it."""
    row = raw_rows.iloc[0].drop(labels=["readmitted"]).to_dict()
    row["prior_encounters"] = 2

    response = client.post("/score", json={"data": row})
    assert response.status_code == 200, response.text

    body = response.json()
    assert 0.0 <= body["readmission_risk_30d"] <= 1.0
    assert body["risk_band"] in {"Low", "Moderate", "High", "Very high"}
    assert isinstance(body["flag_for_followup"], bool)
    assert "clinician review" in body["disclaimer"]


def test_batch_queue_is_ranked_by_risk(raw_rows, client):
    encounters = raw_rows.drop(columns=["readmitted"]).to_dict(orient="records")
    body = client.post("/score/batch", json={"encounters": encounters}).json()

    risks = [item["risk"] for item in body["queue"]]
    assert risks == sorted(risks, reverse=True), "call queue must be highest-risk first"
    assert body["n_flagged"] == sum(item["flag"] for item in body["queue"])


def test_malformed_payload_is_rejected_not_crashed(client):
    response = client.post("/score", json={"data": {"not": "an encounter"}})
    assert response.status_code == 422
    assert "missing required raw columns" in response.json()["detail"]


def test_caller_supplied_history_changes_the_score(raw_rows, client):
    """A single-row request must honour the patient history the EHR passes in.

    Without this the service would score every arriving patient as if they had
    never been admitted before — the strongest feature in the model, silently
    zeroed out.
    """
    base = raw_rows.iloc[0].drop(labels=["readmitted"]).to_dict()

    first_admission = dict(base, prior_encounters=0, number_inpatient=0)
    frequent_flyer = dict(base, prior_encounters=8, number_inpatient=5)

    low = client.post("/score", json={"data": first_admission}).json()
    high = client.post("/score", json={"data": frequent_flyer}).json()

    assert high["readmission_risk_30d"] > low["readmission_risk_30d"]
