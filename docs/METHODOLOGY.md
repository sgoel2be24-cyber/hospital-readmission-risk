# Methodology

How the model was built, why each choice was made, and what would have to be true
for it to run in a hospital.

---

## 1. Framing

The clinical question is not "will this patient be readmitted?" It is:

> A discharge-planning team can call **N** patients a day. Which N?

That reframing changes what counts as success. A model that produces well-ordered
risk scores is useful even if its absolute probabilities are unremarkable,
because the team works down a ranked list. So the pipeline optimises **ranking**
(PR-AUC), fixes an operating point from **capacity** rather than from an
arbitrary 0.5 cutoff, and calibrates at the end so the number a clinician reads
means what it says.

## 2. Preventing leakage

Three leakage paths exist in this dataset. Each is closed and each closure is
tested.

### Patient-level splitting

99,340 encounters belong to 69,987 patients. A row-level split puts the same
patient in train and test, letting the model memorise individuals.

```python
GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    .split(X, y, groups=patient_nbr)
```

Applied twice — once to carve out test, once more to carve validation out of the
remainder — and `GroupKFold` for every cross-validation. Guarded by
`test_split_never_puts_a_patient_on_both_sides`.

### Identifiers excluded from features

`encounter_id` correlates with admission date, and `patient_nbr` would let a tree
memorise individuals outright. Both are removed after the split.
Guarded by `test_identifiers_never_reach_the_feature_matrix`.

### Unscoreable outcomes removed

2,423 encounters end in death or hospice transfer. Those patients cannot be
readmitted, so labelling them "not readmitted" teaches the model that the
features predicting death predict safety.
Guarded by `test_death_and_hospice_encounters_are_removed`.

## 3. Three-way split

```
99,340 encounters
   ├── train       63,605   fit model parameters
   ├── validation  15,962   pick the model, the calibrator, the threshold
   └── test        19,773   read exactly once, at the end
```

Everything that could be tuned is tuned against validation. The test set is not
read until the winner, the calibration method, and the operating threshold are
all frozen — which is what makes the reported 0.684 an estimate of future
performance rather than the best of a dozen peeks.

## 4. Class imbalance

The positive class is 11.39% of rows. Left alone, every model collapses to
predicting "no readmission" for everyone — which scores 88.6% accuracy and is
clinically worthless.

| model | mechanism |
|---|---|
| Logistic regression | `class_weight='balanced'` |
| Random forest | `class_weight='balanced_subsample'` |
| XGBoost | `scale_pos_weight = n_neg / n_pos ≈ 7.8` |
| LightGBM | `class_weight='balanced'` |
| MLP | none available — trained unweighted, judged on ranking |

**SMOTE was tested, not assumed away.** Oversampling inside each CV fold scored
0.2194 PR-AUC versus 0.2200 for class weighting — no better, and it adds a
resampling step to the serving path. Class weighting was kept on the evidence.
See [RESULTS.md](RESULTS.md#ablation-study).

A consequence worth naming: class weighting deliberately distorts the predicted
probabilities upward. That is fine for ranking and wrong for a number a clinician
reads, which is why calibration is a separate step (§7).

## 5. Preprocessing

One `ColumnTransformer`, wrapped with the estimator into a single `Pipeline`, so
the fitted object holds everything needed to score a raw row. Nothing is fitted
outside a fold.

| step | applied to | detail |
|---|---|---|
| `OneHotEncoder` | 20 categorical | `handle_unknown='ignore'`, `min_frequency=20` |
| `StandardScaler` | 19 numeric | logistic regression and MLP only |
| passthrough | 19 numeric | tree models — split points are scale-invariant |

`handle_unknown='ignore'` matters at serving time: a diagnosis group or payer
code never seen in training must not crash the endpoint.

## 6. Model selection

Six models, deliberately spanning the interpretability/capacity range.

1. **Dummy (prior)** — the floor. Anything that cannot beat it on PR-AUC is doing
   nothing.
2. **Logistic regression** — interpretable linear baseline.
3. **Random forest** — bagged, non-linear, no interaction engineering needed.
4. **XGBoost** — gradient boosting, tuned.
5. **LightGBM** — gradient boosting, leaf-wise, tuned.
6. **MLP** — one non-tree, non-linear comparison.

Each is evaluated two ways: **5-fold `GroupKFold` on the training set** (the
selection estimate, with fold-to-fold standard deviation so noise is visible) and
a **single held-out test evaluation** at the end.

Hyperparameters for the two boosters come from a 30-candidate
`RandomizedSearchCV` scored on `average_precision`, 3-fold `GroupKFold`, run on
training patients only ([`src/tune.py`](../src/tune.py), results in
[`reports/best_params.json`](../reports/best_params.json)). Tuning moved XGBoost
from 0.2294 to 0.2352 test PR-AUC.

The winner is chosen on **validation PR-AUC** — not test, and not accuracy.

## 7. Calibration

A class-weighted booster outputs scores that rank well but are systematically too
high. Two calibrators are fitted on validation and compared:

| method | validation Brier | validation PR-AUC |
|---|---|---|
| Isotonic | 0.0954 | 0.2110 |
| **Sigmoid (Platt)** | **0.0959** | **0.2151** |

Isotonic wins Brier by 0.0005 but is a step function — it ties many scores
together and costs 0.004 PR-AUC, which is ranking quality the call list actually
depends on. The selection rule encodes that trade-off: **among calibrators within
0.002 Brier of the best, take the one that ranks better.** Sigmoid wins.

The effect on test: **Brier 0.2007 → 0.0954**, with ROC-AUC and PR-AUC
*unchanged* to four decimal places, because Platt scaling is strictly monotonic
and cannot reorder anything.

## 8. Operating point

A 0.5 threshold is meaningless here. On the calibrated model it flags **zero**
patients, because almost no one has a genuine 50% readmission probability.

The threshold is set from capacity instead:

```python
threshold = quantile(validation_scores, 1 - 0.20)   # flag the top 20%
```

This is the number the hospital actually controls — how many follow-up calls per
day are affordable. The threshold is computed on validation and applied unchanged
to test, where it flags 19.6% of encounters. [`capacity_sweep.png`](../reports/figures/capacity_sweep.png)
shows the full precision/recall trade-off from 5% to 55% capacity so the
threshold can be re-set to a different budget without retraining.

## 9. Explainability

Two views, because they answer different questions.

**Permutation importance** on held-out test data — how much PR-AUC drops when a
feature is shuffled. Model-agnostic, measured on data the model never saw, and it
answers "what does the model actually rely on to generalise?"

**SHAP** (`TreeExplainer`, exact for tree ensembles) — direction and magnitude of
each feature's contribution, both globally and per patient. The per-patient view
is what makes this deployable: a nurse receiving a risk score needs the three or
four facts that produced it, not a number.
[`reports/example_reason_codes.csv`](../reports/example_reason_codes.csv) shows
the format.

Both rank the same features at the top, which is the reassuring outcome.

## 9b. Quantifying uncertainty

Added after the model was frozen. None of it changed the model; two parts of it
corrected claims we were making about the model, which is exactly what this stage
is for.

**Cluster bootstrap** (`src/uncertainty.py`). 2,000 resamples of the test set,
drawing *patients* with replacement and taking all of their encounters. Row-level
resampling would understate the intervals for the same reason a row-level split
overstates performance — encounters within a patient are correlated.

Paired resamples are used for model-vs-model comparisons, so the shared sampling
noise cancels and the interval is on the difference itself. This turned "LightGBM
and XGBoost are indistinguishable" from a judgement into a measurement: Δ PR-AUC
+0.0022, 95% CI [−0.0017, +0.0058].

**Seed sweep** (`src/robustness.py`). The entire result rested on one
`GroupShuffleSplit` with `random_state=42`, and nothing in the project would have
revealed a favourable draw. Re-running the full pipeline under seven seeds showed
seed 42 is the *most* favourable: 0.238 PR-AUC against a cross-seed mean of
0.224 ± 0.010. The seed was fixed before any evaluation, so it is luck rather
than selection — but it is now reported.

**Learning curve** (`src/robustness.py`). Scored on validation, so the "test read
once" guarantee survives. It falsified a claim we had been making: the ablation
study shows seven *methods* converge, which we had described as the information
ceiling of the data. The curve is still rising at full training size (+0.0049
PR-AUC from the last 43% of data). The correct statement is that the ceiling is
on method, not on data.

**Decision curve analysis** (`src/decision_curve.py`). Vickers & Elkin net
benefit across threshold probabilities. Discrimination metrics say the ranking is
good; this says whether *acting* on it beats calling everyone or calling nobody,
across every cost assumption at once, without having to invent a currency figure.

## 10. Fairness

A model can look fine in aggregate while systematically under-serving a subgroup.
Every training run emits per-subgroup flag rate, recall, precision, and ROC-AUC
across `race`, `gender`, and age band
([`reports/fairness_report.csv`](../reports/fairness_report.csv)).

Protected attributes are kept as features rather than dropped. Removing `race`
does not remove its influence — it is reconstructable from other columns — it
only removes the ability to measure the disparity. The gaps this surfaces are
reported honestly in [RESULTS.md](RESULTS.md#fairness); the over-80 result is the
one that would block deployment.

Two further checks (`src/equity.py`):

**Per-subgroup calibration**, not just discrimination. A risk score can be well
calibrated overall while systematically misleading for one group. Gender and the
two large racial groups come out near-perfect; patients over 80 have a
calibration slope of 0.79, corroborating the discrimination finding from an
independent angle.

**Age-stratified thresholds**, tested rather than assumed. Giving each age band
its own capacity threshold costs 10 true positives out of 901 and moves calls
away from the oldest band toward the 40–60 band. It does not fix the underlying problem,
because thresholds reallocate capacity and cannot change ranking — and since
the oldest band have the highest base rate in the data, whether the reallocation is
*fairer* is a value judgement rather than a metric. Documented as a measured
trade-off, not a solution.

## 11. Deployment

**Positioning.** Discharge-planning triage. Score every discharge, rank them, call
the top 20% within 7 days. Decision support requiring human review — never an
autonomous action, and never a reason to withhold care from an unflagged patient.

**Serving.** The whole model is one 5 MB joblib artefact holding the
preprocessing, the booster, the calibrator, and the threshold. Tree inference is
sub-millisecond; the bottleneck is the database read, not the model.

- `GET /health` — deployed model, calibration method, threshold
- `POST /score` — one encounter → risk, flag, band, disclaimer
- `POST /score/batch` — nightly discharge list → ranked call queue

**Two integration modes.** Nightly batch (`src/predict.py`) writing a ranked list
into the discharge-planning worklist, or a real-time widget in an existing
dashboard (`src/api.py`).

**The training/serving skew that had to be fixed.** Two bugs in the serving path
were found and closed while building the API, both invisible to offline metrics:

1. Feature engineering required the `readmitted` label — which does not exist at
   discharge. The target is now built only when the column is present.
2. `prior_encounters` was derived by counting within the batch, so a single-row
   request scored every patient as a first-ever admission — zeroing out the
   model's strongest feature. The EHR can now supply the count directly.

Both are covered by `tests/test_serving.py`. Serving reuses `data_prep.py`
directly rather than reimplementing it, because a second copy of the feature
logic is how a model that scored well offline quietly degrades in production.

**Monitoring.** The data is from 1999–2008; coding practice, medications, and
discharge policy have all moved since. Before use: recalibrate on recent local
data, then monitor the flag rate (drift shows up here first), the realised
readmission rate among flagged patients versus predicted, calibration drift, and
the subgroup gaps in §10 — on a schedule, not once at launch.

**What this model is not.** It is not a diagnosis, not a discharge-readiness
assessment, and not evidence that an unflagged patient is safe. 59.8% of
readmissions are not flagged at 20% capacity. It reprioritises attention; it does
not replace it.
