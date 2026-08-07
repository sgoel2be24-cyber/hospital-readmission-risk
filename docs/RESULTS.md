# Results

All numbers are on the **held-out test set** — 19,773 encounters from 13,998
patients who appear nowhere in training or validation. Base rate 11.35%.

Reproduce with `make train`. Raw values in
[`reports/metrics.json`](../reports/metrics.json).

---

## Headline

| metric | value | no-skill reference |
|---|---|---|
| **PR-AUC** | **0.238** | 0.113 (base rate) |
| **ROC-AUC** | **0.684** | 0.500 |
| **Brier** | **0.095** | 0.101 |
| Recall @ 20% capacity | **40.2%** | 20% |
| Precision @ 20% capacity | **23.3%** | 11.3% |
| Lift over base rate | **2.05×** | 1.00× |

Calling the 20% of discharges the model ranks highest catches **40.2% of all
30-day readmissions**. A flagged patient is **2.05× more likely** to return than
an average one.

## Why accuracy is not the headline

| | accuracy | readmissions caught |
|---|---|---|
| This model @ deployed threshold | 78.2% | 40.2% |
| "Nobody will be readmitted" | **88.6%** | **0%** |
| Best accuracy at any threshold | 88.8% | 0.4% of patients flagged |

With an 11% positive rate, **accuracy rewards inaction**. The do-nothing model
beats ours by ten points and delivers nothing. Optimising accuracy on this
problem means optimising toward a model that never flags anyone.

This is also why the deployed threshold is 0.166, not 0.5. At 0.5 the calibrated
model flags **zero** patients — almost no discharge carries a genuine 50%
readmission probability. Both are in `metrics.json` as `at_capacity` and `at_0.5`.

Accuracy is reported for completeness. Nothing was selected using it.

## Comparative model analysis

Cross-validation is 5-fold `GroupKFold` on training patients; test is the single
final evaluation.

| model | CV ROC-AUC | CV PR-AUC | test ROC-AUC | test PR-AUC | test Brier | recall @20% | lift | fit (s) |
|---|---|---|---|---|---|---|---|---|
| Dummy (prior) | 0.500 ±0.000 | 0.114 ±0.001 | 0.500 | 0.114 | 0.101 | — | 1.00× | 2.8 |
| Logistic regression | 0.658 ±0.006 | 0.204 ±0.006 | 0.667 | 0.213 | 0.227 | 35.8% | 1.85× | 4.2 |
| Random forest | 0.670 ±0.005 | 0.211 ±0.006 | 0.679 | 0.225 | 0.197 | 38.8% | 1.97× | 19.4 |
| XGBoost | 0.674 ±0.006 | 0.220 ±0.006 | **0.684** | 0.235 | 0.216 | 39.8% | 2.04× | 10.4 |
| **LightGBM** ⬅ | **0.676 ±0.006** | **0.220 ±0.007** | 0.684 | **0.238** | 0.201 | **40.2%** | **2.05×** | 16.8 |
| MLP | 0.653 ±0.006 | 0.202 ±0.008 | 0.665 | 0.222 | 0.096 | 36.3% | 1.84× | 6.6 |

![model comparison](../reports/figures/model_comparison.png)

**Reading this table honestly.** LightGBM is selected, but XGBoost is inside noise
of it — 0.0003 ROC-AUC apart, against a fold-to-fold standard deviation of 0.006.
The defensible claim is *"gradient boosting beats linear and bagged models by
about 0.02 PR-AUC; the two boosters are indistinguishable."* Selection went to
LightGBM on validation PR-AUC.

Other observations worth naming:

- **The ordering is stable across CV and test**, and test sits consistently
  ~0.01 above CV. That gap is expected — CV models train on 80% of the training
  set, the final models on all of it.
- **MLP has the best Brier (0.096) while ranking worst.** It is the only model
  trained *unweighted*, so its probabilities are undistorted. This is a clean
  demonstration that calibration and discrimination are separate properties.
- **Logistic regression reaches 0.213 PR-AUC** — 89% of the boosted result from a
  fully interpretable model. If interpretability were a hard constraint, the cost
  of that constraint is about 0.025 PR-AUC.

![ROC curves](../reports/figures/roc_curves.png)
![PR curves](../reports/figures/pr_curves.png)

## Calibration

Class weighting is what makes ranking work and probabilities wrong. Platt scaling
fixes the probabilities without touching the ranking.

| | Brier | ROC-AUC | PR-AUC |
|---|---|---|---|
| LightGBM, raw | 0.2007 | 0.6839 | 0.2376 |
| **LightGBM + Platt** | **0.0954** | 0.6839 | 0.2376 |

Brier more than halves; discrimination is bit-identical, because a strictly
monotonic transform cannot reorder anything. Isotonic regression scored a
marginally better Brier on validation (0.0954 vs 0.0959) but cost 0.004 PR-AUC by
tying scores together, so it was rejected.

![calibration](../reports/figures/calibration.png)

The uncalibrated curves sitting well below the diagonal are the class-weighting
distortion made visible: those models predict ~0.5 for patients who are readmitted
about 15% of the time. After calibration the curve tracks the diagonal — a
predicted 18% means roughly 18%.

## Operating point

At the deployed threshold of 0.166, flagging 19.6% of encounters:

| | flagged | not flagged |
|---|---|---|
| **Readmitted <30d** | 901 (TP) | 1,343 (FN) |
| **Not readmitted** | 2,965 (FP) | 14,564 (TN) |

Precision 23.3% · Recall 40.2% · Specificity 83.1% · F1 0.295

Read operationally: of 100 calls, ~23 reach a patient who would have been
readmitted. Against a base rate of 11.3%, the call list is worth **2.05× a random
one**.

![capacity sweep](../reports/figures/capacity_sweep.png)

The threshold is a business decision, not a model property. The curve above prices
each option:

| capacity | recall | precision |
|---|---|---|
| 10% | 24.4% | 27.7% |
| **20%** | **40.6%** | **23.1%** |
| 30% | 52.9% | 20.0% |
| 40% | 63.9% | 18.1% |
| 50% | 72.1% | 16.4% |

(These use thresholds taken at test-set quantiles, so they differ marginally from
the 40.2% / 23.3% reported above, where the threshold is fixed on validation and
applied blind to test. The deployed figure is the honest one; this table is for
comparing capacity options against each other.)

Doubling capacity from 20% to 40% buys **23 more percentage points of recall** for
5 points of precision. Whether that trade is worth it depends on the cost of a
call versus the cost of a readmission — a hospital finance question, and the model
supports either answer without retraining.

## Ablation study

Seven variants, all scored with identical 5-fold `GroupKFold` on training
patients. The test set was not read during any of this.

| variant | features | CV PR-AUC | Δ PR-AUC | CV ROC-AUC | Δ ROC-AUC | adopted |
|---|---|---|---|---|---|---|
| **A — current** | 39 | **0.2200 ±0.0069** | — | **0.6762** | — | ✅ baseline |
| B — drop `payer_code` | 38 | 0.2180 | −0.0020 | 0.6736 | −0.0026 | ❌ |
| C — +10 engineered features | 49 | 0.2196 | −0.0005 | 0.6749 | −0.0013 | ❌ |
| D — B and C combined | 48 | 0.2178 | −0.0023 | 0.6730 | −0.0032 | ❌ |
| E — native categorical splits | 39 | 0.2194 | −0.0007 | 0.6753 | −0.0009 | ❌ |
| F — soft-vote RF+XGB+LGBM | 39 | 0.2191 | −0.0009 | 0.6760 | −0.0002 | ❌ |
| G — SMOTE not class weighting | 39 | 0.2194 | −0.0006 | 0.6740 | −0.0022 | ❌ |

**Nothing beat the baseline.** Every delta is negative and every one is smaller
than the fold-to-fold standard deviation of ±0.0069 — so the honest reading is
"no measurable difference", not "slightly worse". No variant was adopted.

Three results here are worth more than the arithmetic:

- **SMOTE (G) does not beat class weighting.** The brief names both. This settles
  it with a measurement rather than a preference, and class weighting has the
  further advantage of adding nothing to the serving path.
- **Ensembling (F) buys nothing.** The three tree models make correlated errors —
  they are reading the same limited signal, not complementary ones.
- **Dropping `payer_code` (B) costs 0.002 PR-AUC.** It is not free, so keeping it
  is a real trade rather than laziness. See [Limitations](#limitations).

Convergence of seven independent approaches on 0.22 PR-AUC is itself the finding:
this is the information ceiling of administrative billing data for this outcome,
not an under-tuning problem. Reproduce with `make experiments`.

## What drives the predictions

Permutation importance (drop in test PR-AUC when shuffled) and SHAP agree on the
top of the list.

| rank | feature | permutation Δ PR-AUC |
|---|---|---|
| 1 | `number_inpatient` | 0.0360 |
| 2 | `discharge_group` | 0.0328 |
| 3 | `prior_encounters` | 0.0107 |
| 4 | `payer_code` | 0.0101 |
| 5 | `diag_1_group` | 0.0096 |
| 6 | `diag_3_group` | 0.0053 |
| 7 | `age_mid` | 0.0046 |

![SHAP](../reports/figures/shap_beeswarm.png)

Clinically, this is the reassuring result:

- **Prior inpatient admissions dominate.** Patients who have been admitted before
  are readmitted again. Utilisation history beats anything measured during the
  stay.
- **Where the patient goes matters.** Discharge to home lowers risk; discharge to
  a care facility raises it — a proxy for frailty and dependency.
- **Primary diagnosis matters, circulatory most.** Heart failure and related
  conditions push risk up; respiratory pushes it down.
- **Nothing suspicious sits at the top.** No identifier, no date artefact — which
  is what you want to see after closing three leakage paths.

`reports/example_reason_codes.csv` shows the per-patient form the ward would
receive:

> Risk 0.43 — `number_inpatient=16` (+0.996); `prior_encounters=22` (+0.401);
> `prior_visits_total=19` (+0.261)

## Fairness

Per-subgroup performance at the deployed threshold
([`reports/fairness_report.csv`](../reports/fairness_report.csv)). Groups under
200 test encounters are suppressed.

### By race

| group | n | base rate | flag rate | recall | precision | ROC-AUC |
|---|---|---|---|---|---|---|
| Caucasian | 14,813 | 11.7% | 19.8% | 40.4% | 23.9% | 0.684 |
| African American | 3,697 | 10.8% | 20.2% | 40.1% | 21.4% | 0.670 |
| Hispanic | 398 | 9.1% | 16.8% | 47.2% | 25.4% | 0.801 |
| Other | 302 | 10.3% | 12.9% | 32.3% | 25.6% | 0.684 |
| Unknown | 447 | 7.2% | 14.1% | 31.3% | 15.9% | 0.658 |

### By gender

| group | n | base rate | flag rate | recall | precision | ROC-AUC |
|---|---|---|---|---|---|---|
| Female | 10,726 | 11.4% | 20.1% | 42.4% | 24.2% | 0.694 |
| Male | 9,047 | 11.2% | 19.0% | 37.5% | 22.2% | 0.673 |

### By age

| group | n | base rate | flag rate | recall | precision | ROC-AUC |
|---|---|---|---|---|---|---|
| <40 | 1,289 | 12.2% | 20.2% | 58.0% | 34.9% | 0.784 |
| 40–60 | 5,376 | 10.4% | 14.8% | 39.6% | 27.9% | 0.712 |
| 60–75 | 9,277 | 11.4% | 20.8% | 39.7% | 21.8% | 0.673 |
| **75+** | **3,831** | **12.2%** | **23.0%** | **35.8%** | **19.0%** | **0.613** |

**The finding that matters is age, not race.** Race and gender gaps are modest —
the two large racial groups get near-identical flag rates and recall.

The over-75 group is the problem. It has the **highest flag rate (23.0%)**, the
**lowest precision (19.0%)**, the **lowest recall (35.8%)**, and by far the
**worst discrimination (ROC-AUC 0.613)**. The model flags the most elderly
patients while ranking them worst — it is spending disproportionate follow-up
capacity on the group where its ordering is least trustworthy. Since almost
everything the model relies on is a proxy for frailty, and nearly all patients
over 75 look frail, the features stop separating within that group.

**This would have to be addressed before deployment.** Options: a separate model
for over-75s, an age-stratified threshold, or restricting the tool to under-75s
and triaging older patients by existing clinical judgement. It is not fixable by
tuning.

## Limitations

**1. Discrimination is modest, and that is the ceiling.** ROC-AUC 0.684 sits in
the 0.64–0.70 band published for this dataset and target. Seven approaches
converged on it. A materially higher number on this data almost always indicates
a leaked split rather than a better model.

**2. `payer_code_Unknown` is a top-6 driver.** Insurance coding is not
physiology — it is very likely a proxy for hospital, era, or documentation
practice. It generalises within this dataset and may not generalise to a new
hospital. Removing it costs 0.002 PR-AUC (ablation B), so it is retained
knowingly, and it is the first thing to re-examine on local data.

**3. The data is 1999–2008.** Coding standards, drug availability, and discharge
policy have all changed. Recalibration on recent local data is mandatory, not
optional.

**4. Diabetic inpatients only, 1–14 day stays.** The model does not transfer to
general medical populations without revalidation.

**5. 59.8% of readmissions are missed at 20% capacity.** This tool reprioritises
attention. It does not identify every at-risk patient, and an unflagged patient
is not a safe patient.

**6. Readmissions to other hospitals are invisible.** The dataset records only
returns within the same network, so the true positive rate is understated and
the model is trained on an incomplete label.

## Reproducibility

Every number above regenerates from a clean clone:

```bash
make setup && make data && make tune && make train && make experiments && make explain && make test
```

Fixed seed (42) throughout. Runtime: ~60 s for `train`, ~5 min for `tune`, ~2 min
for `experiments`. 17 tests cover ICD-9 grouping, label construction, all three
leakage paths, split determinism, feature consistency, and the serving path.
