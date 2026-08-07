"""Central configuration: paths, column groups, and modelling constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_CSV = DATA_DIR / "diabetic_data.csv"
IDS_MAPPING_CSV = DATA_DIR / "IDS_mapping.csv"

for _d in (MODEL_DIR, REPORT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of the train split held out for threshold/calibration tuning
CV_FOLDS = 5

# Fraction of discharges the follow-up team can realistically call. The decision
# threshold is set so that this share of patients gets flagged.
CAPACITY_FRACTION = 0.20

# discharge_disposition_id values meaning the patient died or went to hospice:
# these encounters cannot produce a 30-day readmission and would bias the label.
DEAD_OR_HOSPICE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

# Columns dropped outright.
#   weight              -> ~97% missing
#   payer_code          -> billing artefact, kept as "Unknown" category instead (see data_prep)
#   examide/citoglipton -> single constant value, zero information
#   encounter_id        -> identifier
#   patient_nbr         -> grouping key, removed after the split
ID_COLS = ["encounter_id", "patient_nbr"]
DROP_COLS = ["weight", "examide", "citoglipton"]

# The 23 medication columns with values in {No, Steady, Up, Down}.
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Medications kept as individual categorical features. The rest are almost
# entirely "No" and are summarised by the engineered counts instead.
KEY_MEDICATION_COLS = ["metformin", "insulin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone"]

TARGET_COL = "readmitted"
POSITIVE_LABEL = "<30"
