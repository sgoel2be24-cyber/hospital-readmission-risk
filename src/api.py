"""Real-time scoring service — the online half of deployment.

    uvicorn src.api:app --reload --port 8000
    curl -X POST localhost:8000/score -H 'content-type: application/json' -d @sample_request.json

The model is a decision-support tool: the response carries a risk score, a
follow-up flag derived from the capacity threshold fixed at training time, and
the drivers behind the score. It never states a decision on its own.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import load_artefact, score

app = FastAPI(
    title="30-Day Readmission Risk — discharge triage",
    description="Decision support for discharge planning. Requires clinician review.",
    version="1.0.0",
)


class Encounter(BaseModel):
    """A raw encounter row, exactly as it comes out of the hospital EHR export."""

    data: dict[str, Any] = Field(
        ...,
        description="Column -> value map matching the source dataset schema "
        "(race, gender, age, admission_type_id, diag_1, insulin, ...). Include "
        "`prior_encounters` — this patient's earlier admission count from the EHR — "
        "otherwise a single-row request is scored as a first-ever admission.",
    )


class BatchRequest(BaseModel):
    encounters: list[dict[str, Any]]


@app.get("/health")
def health() -> dict:
    artefact = load_artefact()
    return {
        "status": "ok",
        "model": artefact["model_name"],
        "calibration": artefact["calibration"],
        "threshold": round(artefact["threshold"], 4),
        "flags_top_fraction": artefact["capacity_fraction"],
    }


@app.post("/score")
def score_one(encounter: Encounter) -> dict:
    artefact = load_artefact()
    try:
        result = score(pd.DataFrame([encounter.data]), artefact)
    except Exception as exc:  # malformed payload -> 422, not a 500
        raise HTTPException(status_code=422, detail=f"could not score encounter: {exc}") from exc
    if result.empty:
        # clean() drops death/hospice discharges — those patients are not scoreable.
        raise HTTPException(status_code=422, detail="encounter is not eligible for scoring "
                                                    "(death or hospice discharge)")
    row = result.iloc[0]
    return {
        "readmission_risk_30d": round(float(row["readmission_risk_30d"]), 4),
        "flag_for_followup": bool(row["flag_for_followup"]),
        "risk_band": str(row["risk_band"]),
        "disclaimer": "Decision support only. Requires clinician review; not an autonomous action.",
    }


@app.post("/score/batch")
def score_batch(request: BatchRequest) -> dict:
    """Nightly discharge list -> ranked call queue for the follow-up team."""
    artefact = load_artefact()
    try:
        result = score(pd.DataFrame(request.encounters), artefact)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not score batch: {exc}") from exc
    result = result.sort_values("readmission_risk_30d", ascending=False)
    return {
        "n_scored": int(len(result)),
        "n_flagged": int(result["flag_for_followup"].sum()),
        "queue": [
            {
                "index": int(i),
                "risk": round(float(r["readmission_risk_30d"]), 4),
                "band": str(r["risk_band"]),
                "flag": bool(r["flag_for_followup"]),
            }
            for i, r in result.iterrows()
        ],
    }
