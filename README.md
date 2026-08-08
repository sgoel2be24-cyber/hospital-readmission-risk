# Hospital 30-Day Readmission Risk Prediction

Predicting which patients will be readmitted within 30 days of discharge, so a
hospital can point its limited follow-up capacity at the people most likely to
come back.

**ML Bubble 2026** — Track TE-BE, Design & Solve (Advanced)

---

## The problem

Roughly one in nine diabetic hospital discharges results in a readmission within
30 days. Many of those are preventable with a phone call, a medication check, or
an earlier follow-up appointment — but a discharge-planning team can only reach a
fraction of patients per day. The question is not "who might come back?", it is
**"given that we can call 20% of today's discharges, which 20%?"**

That framing drives every modelling decision in this repository.

## Result

Scoring every discharge and calling the highest-risk 20%:

| | Model | Call 20% at random |
|---|---|---|
| Readmissions caught | **40.2%** | 20% |
| Precision of the call list | **23.3%** | 11.3% |
| Lift over base rate | **2.05×** | 1.00× |

On the held-out test set of 19,773 encounters: **ROC-AUC 0.684** (95% CI
0.671–0.698), **PR-AUC 0.238** (0.213–0.265) against a base rate of 0.113,
**Brier 0.095** after calibration. Intervals are a cluster bootstrap over
patients.

Across seven random splits the same pipeline averages **ROC-AUC 0.678 ± 0.006**
and **PR-AUC 0.224 ± 0.010** — the reported split is the most favourable of the
seven, and [docs/RESULTS.md](docs/RESULTS.md#robustness) says so rather than
burying it.

Decision curve analysis puts it concretely: **15.6 extra readmissions caught per
1,000 discharges**, net of the false alarms they cost, and the model beats both
"call everyone" and "call nobody" across every plausible cost assumption.

> **A note on accuracy.** This model is 78.2% accurate. A model that predicts "no
> readmission" for everyone is 88.6% accurate and catches zero readmissions. With
> an 11% positive rate, accuracy rewards doing nothing — which is exactly why it
> is not the metric used to select anything here. See [docs/RESULTS.md](docs/RESULTS.md).

## Quickstart

```bash
make setup && make data && make train
```

`make train` runs in about 60 seconds and writes the trained model to `models/`
plus every metric and figure to `reports/`.

| command | what it does |
|---|---|
| `make setup` | create `.venv`, install `requirements.txt` |
| `make data` | download the UCI dataset (18 MB) into `data/` |
| `make tune` | randomised hyperparameter search (~5 min) |
| `make train` | train + compare all 6 models, calibrate, write reports |
| `make experiments` | 7-variant ablation study (~2 min) |
| `make explain` | permutation importance + SHAP + reason codes |
| `make evidence` | bootstrap CIs, seed sweep, learning curve, equity, decision curve |
| `make test` | 26 tests |
| `make api` | serve the scoring API on `:8000` |
| `make score` | batch-score every encounter to CSV |

<details>
<summary>macOS: <code>libomp</code> is required for XGBoost and LightGBM</summary>

Both ship macOS wheels that link against OpenMP but do not bundle it. If
`import xgboost` fails with `Library not loaded: @rpath/libomp.dylib`:

```bash
brew install libomp
```

Linux and Google Colab are unaffected — the wheels there are self-contained.
</details>

## What's here

```
src/config.py        paths, column groups, constants
src/data_prep.py     cleaning, ICD-9 grouping, feature engineering
src/pipeline.py      preprocessing + the 6-model zoo
src/tune.py          randomised hyperparameter search
src/train.py         train, compare, calibrate, fairness, figures
src/evaluate.py      metrics, operating point, subgroup report, plots
src/experiments.py   ablation study — every variant we tried and rejected
src/explain.py       permutation importance, SHAP, per-patient reason codes
src/uncertainty.py   cluster-bootstrap CIs, incl. paired model comparisons
src/robustness.py    seed sweep + learning curve
src/equity.py        age-stratified thresholds, per-subgroup calibration
src/decision_curve.py  net benefit vs call-everyone / call-nobody
src/predict.py       batch scoring CLI
src/api.py           FastAPI service
tests/               26 tests: leakage guards, serving path, bootstrap and
                     net-benefit maths
```

## Documentation

| document | contents |
|---|---|
| [docs/DATASET.md](docs/DATASET.md) | source, schema, cleaning decisions, feature engineering |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | splitting, imbalance, model selection, calibration, deployment |
| [docs/RESULTS.md](docs/RESULTS.md) | full metrics, comparative analysis, ablation, fairness, limitations |

Generated artefacts live in `reports/`: [metrics.json](reports/metrics.json),
[model_comparison.csv](reports/model_comparison.csv),
[ablation_study.csv](reports/ablation_study.csv),
[bootstrap_ci.json](reports/bootstrap_ci.json),
[robustness.json](reports/robustness.json),
[equity.json](reports/equity.json),
[decision_curve.json](reports/decision_curve.json),
[fairness_report.csv](reports/fairness_report.csv), and thirteen figures in
`reports/figures/`.

## The one decision that matters most

Some patients appear in this dataset many times — 99,340 encounters belong to
only 69,987 patients. Splitting train/test **by row** puts the same patient on
both sides, letting the model memorise individuals and inflating every score.
Splitting **by `patient_nbr`** is the difference between an honest 0.68 and a
meaningless 0.85.

```python
GroupShuffleSplit(...).split(X, y, groups=patient_nbr)
```

`tests/test_pipeline.py::test_split_never_puts_a_patient_on_both_sides` fails the
build if that ever regresses.

## Deployment

The model is decision support, not an autonomous action. It produces a ranked
call list; a clinician decides what to do with it.

```bash
make api
curl -s localhost:8000/health
```

`POST /score` returns a calibrated risk, a follow-up flag, a risk band, and a
disclaimer. `POST /score/batch` turns a nightly discharge list into a ranked
queue. Inference is sub-millisecond; the whole service is one 5 MB artefact.

Before any real use, the subgroup gaps in
[docs/RESULTS.md](docs/RESULTS.md#fairness) need addressing — most notably that
discrimination degrades for patients over 75, the group being flagged most often.
