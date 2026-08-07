# Dataset Details

## Source

**Diabetes 130-US hospitals for years 1999–2008**
UCI Machine Learning Repository, dataset 296 —
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Ten years of clinical care at 130 US hospitals and integrated delivery networks.
Every row is one inpatient encounter for a patient with diabetes listed as a
diagnosis, where the stay lasted 1–14 days, laboratory tests were performed, and
medications were administered.

Downloaded reproducibly with `make data`. The raw CSVs are not committed (18 MB);
the download is a single `curl` in the [Makefile](../Makefile).

| property | value |
|---|---|
| Raw encounters | 101,766 |
| Raw columns | 50 |
| Unique patients | 71,518 |
| Encounters after cleaning | 99,340 |
| Patients after cleaning | 69,987 |
| Features used for modelling | 39 |
| Positive rate (`<30` readmission) | 11.39% |

The gap between 101,766 encounters and 71,518 patients is the single most
important fact about this dataset — see [Repeat patients](#repeat-patients).

## Target definition

The raw `readmitted` column has three values:

| value | meaning | count | share |
|---|---|---|---|
| `NO` | no readmission recorded | 54,864 | 53.9% |
| `>30` | readmitted after 30 days | 35,545 | 34.9% |
| `<30` | **readmitted within 30 days** | 11,357 | 11.2% |

Binarised to **`<30` versus everything else**. This is the standard formulation
for this dataset and the clinically meaningful one: 30 days is the window used by
readmission-penalty programmes, and it is short enough that the readmission is
plausibly connected to the quality of the discharge.

Collapsing `>30` into the negative class is deliberate. A readmission fourteen
months later is not a discharge-planning failure, and treating it as a positive
would dilute the signal the follow-up team needs.

## Repeat patients

`encounter_id` is unique per row. `patient_nbr` is not — the median patient
appears once, 16,773 patients appear more than once, and the most frequent single
patient accounts for **40 encounters**.

```
99,340 encounters  →  69,987 distinct patients
```

Splitting train/test by row lets the same patient land on both sides. The model
then partly memorises individuals rather than learning transferable risk, and
every metric comes out optimistic. All splits in this project are on
`patient_nbr` via `GroupShuffleSplit`, and cross-validation uses `GroupKFold`.

This is enforced by a test, not by convention:
`tests/test_pipeline.py::test_split_never_puts_a_patient_on_both_sides`.

## Cleaning decisions

Implemented in [`src/data_prep.py`](../src/data_prep.py).

### Rows removed — 2,426 total (2.4%)

| filter | rows | why |
|---|---|---|
| Death or hospice discharge | 2,423 | `discharge_disposition_id` in {11, 13, 14, 19, 20, 21}. These patients **cannot** be readmitted. Leaving them in labels them as negatives and teaches the model that dying is a good outcome. |
| `gender = Unknown/Invalid` | 3 | unusable value |

No rows are dropped for missing values.

### Columns removed

| column | why |
|---|---|
| `weight` | ~97% missing — nothing to impute from |
| `examide`, `citoglipton` | single constant value across all 101,766 rows; zero information |
| `encounter_id`, `patient_nbr` | identifiers; `patient_nbr` is retained as the split key but never as a feature |

### Missing values

The token in this dataset is `?`, not `NaN`. It is treated as an **explicit
category**, never imputed and never dropped:

| column | `?` share | handling |
|---|---|---|
| `race` | 2.2% | → `Unknown` |
| `payer_code` | 39.6% | → `Unknown` |
| `medical_specialty` | 49.1% | → `Unknown` |

Missingness here is administrative rather than random — an unrecorded payer or
admitting specialty says something about how the encounter was documented, and
that pattern turns out to carry signal. `payer_code_Unknown` is the sixth-largest
SHAP driver in the final model, which is discussed as a limitation in
[RESULTS.md](RESULTS.md#limitations).

One subtle trap: `max_glu_serum` and `A1Cresult` use the literal string `"None"`
to mean *the test was never ordered*. Pandas converts that to `NaN` by default,
silently destroying a genuinely informative level — a stay where HbA1c was never
checked is different from one where it was checked and came back normal. The
loader passes `keep_default_na=False` and renames the level to `Not_measured`.

## Feature engineering

39 features from 47 usable raw columns.

### Diagnoses: 2,000+ ICD-9 codes → 9 categories

`diag_1`, `diag_2`, `diag_3` carry 717, 749, and 790 distinct ICD-9 codes. One-hot
encoding them raw produces thousands of near-empty columns. They are grouped into
chapter-level categories following the structure used in the original published
study of this dataset:

| category | ICD-9 range |
|---|---|
| Circulatory | 390–459, 785 |
| Respiratory | 460–519, 786 |
| Digestive | 520–579, 787 |
| Diabetes | 250.xx |
| Injury | 800–999, E-codes |
| Musculoskeletal | 710–739 |
| Genitourinary | 580–629, 788 |
| Neoplasms | 140–239 |
| Other | everything else, including V-codes |

### Administrative codes → interpretable groups

`discharge_disposition_id` (26 levels) → `Home`, `Home_health`, `Care_facility`,
`Another_hospital`, `Left_AMA`, `Unknown`, `Other`. Similar grouping for
`admission_type_id` and `admission_source_id`. This both reduces sparsity and
makes the SHAP output readable by someone who does not know the code tables.

### Ordinal age

`age` arrives as brackets like `[70-80)`. One-hot encoding throws away the
ordering; the bracket midpoint (`75.0`) keeps it in one column.

### Derived features

| feature | definition | rationale |
|---|---|---|
| `prior_visits_total` | outpatient + emergency + inpatient | total prior utilisation |
| `has_prior_inpatient` | `number_inpatient > 0` | crossing zero matters more than the count |
| `has_prior_emergency` | `number_emergency > 0` | as above |
| `prior_encounters` | count of this patient's earlier encounters | history depth; known at discharge, and patients never straddle the split |
| `n_meds_active` | medications not `No` (of 21) | regimen complexity |
| `n_med_changes` | medications `Up` or `Down` | regimen instability during the stay |
| `n_meds_increased` / `n_meds_decreased` | direction of change | escalation vs de-escalation |
| `meds_per_day` | `num_medications / time_in_hospital` | intensity, not just duration |
| `procedures_per_day` | `(num_procedures + num_lab_procedures) / days` | as above |

The 21 medication columns are collapsed to the four counts above plus six
individually-retained drugs (metformin, insulin, glipizide, glyburide,
pioglitazone, rosiglitazone). The remainder are prescribed so rarely that their
one-hot columns are almost entirely zero.

Rare levels in `medical_specialty` and `payer_code` (under 1% frequency) are
collapsed to `Other`.

### Features that were built and rejected

Ten further features — comorbidity spread, HbA1c-tested-and-changed interaction,
history concentration ratios, lab intensity — were built, measured, and **not
adopted**: they cost 0.0005 CV PR-AUC. The code remains in `data_prep.py` behind
`build_dataset(include_extra=True)` so the result is reproducible, and a test
prevents them re-entering the default feature set. See
[RESULTS.md](RESULTS.md#ablation-study).

## Final feature matrix

39 features → 132 columns after one-hot encoding (rare levels below 20
occurrences are folded into an `infrequent` bucket by the encoder).

**19 numeric:** `time_in_hospital`, `num_lab_procedures`, `num_procedures`,
`num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`,
`number_diagnoses`, `age_mid`, `prior_visits_total`, `has_prior_inpatient`,
`has_prior_emergency`, `prior_encounters`, `n_meds_active`, `n_med_changes`,
`n_meds_increased`, `n_meds_decreased`, `meds_per_day`, `procedures_per_day`

**20 categorical:** `race`, `gender`, `payer_code`, `medical_specialty`,
`max_glu_serum`, `A1Cresult`, `metformin`, `glipizide`, `glyburide`,
`pioglitazone`, `rosiglitazone`, `insulin`, `change`, `diabetesMed`,
`diag_1_group`, `diag_2_group`, `diag_3_group`, `discharge_group`,
`admission_type_group`, `admission_source_group`

## Split

| split | encounters | patients | positive rate |
|---|---|---|---|
| Train | 63,605 | 44,791 | 11.42% |
| Validation | 15,962 | 11,198 | 11.31% |
| Test | 19,773 | 13,998 | 11.35% |

Validation exists so that the model choice, the calibrator, and the decision
threshold are all fixed **before** the test set is read once, at the end.

## Ethical notes

The dataset carries `race`, `gender`, and `age`. They are retained as features
because excluding a protected attribute does not remove its influence — it only
removes the ability to measure it. A per-subgroup performance report is generated
on every training run ([`reports/fairness_report.csv`](../reports/fairness_report.csv))
and the gaps it exposes are documented in [RESULTS.md](RESULTS.md#fairness).

The data is de-identified and publicly released for research. `patient_nbr` is a
surrogate key, used only for grouping and never as a model input.
