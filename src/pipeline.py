"""Preprocessing pipeline and the model zoo used for the comparative analysis."""

from __future__ import annotations

import json

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE, REPORT_DIR

TUNED_PARAMS_PATH = REPORT_DIR / "best_params.json"


def load_tuned_params(model_name: str) -> dict:
    """Hyperparameters written by `python -m src.tune`, if that step has been run."""
    if not TUNED_PARAMS_PATH.exists():
        return {}
    return json.loads(TUNED_PARAMS_PATH.read_text()).get(model_name, {})


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """One-hot encode categoricals; scale numerics only for the models that need it.

    `scale=True` for the linear model and the MLP (gradient-based, scale sensitive),
    `scale=False` for the tree ensembles (split points are scale invariant).
    """
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(y_train: np.ndarray) -> dict[str, Pipeline]:
    """Return {name: unfitted estimator}, ordered from interpretable to complex.

    Every model handles the ~11% positive rate explicitly: `class_weight='balanced'`
    for the sklearn estimators, `scale_pos_weight` for XGBoost. Without this the
    models collapse to predicting "no readmission" for everyone. MLPClassifier is
    the exception — it has no class-weight hook, so it is trained unweighted and
    judged on ranking metrics, where the imbalance matters less.
    """
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    pos_weight = n_neg / max(n_pos, 1)

    xgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
    )
    xgb_params.update(load_tuned_params("xgboost"))

    lgb_params = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
    )
    lgb_params.update(load_tuned_params("lightgbm"))

    return {
        # Reference point: predicts the base rate for everyone. Any model that
        # cannot beat this on PR-AUC is doing nothing useful.
        "baseline_majority": (
            DummyClassifier(strategy="prior"),
            False,
        ),
        "logistic_regression": (
            LogisticRegression(
                max_iter=3000,
                C=0.1,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=14,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "xgboost": (
            XGBClassifier(
                **xgb_params,
                scale_pos_weight=pos_weight,
                tree_method="hist",
                eval_metric="aucpr",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "lightgbm": (
            LGBMClassifier(
                **lgb_params,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            False,
        ),
        "mlp": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-3,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=60,
                early_stopping=True,
                n_iter_no_change=5,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    }


def assemble(model_spec, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Wrap an estimator with its preprocessing into a single deployable object."""
    estimator, needs_scaling = model_spec
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric, categorical, scale=needs_scaling)),
            ("model", estimator),
        ]
    )
