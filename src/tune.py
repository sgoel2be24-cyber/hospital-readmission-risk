"""Randomised hyperparameter search for the two gradient-boosting models.

    python -m src.tune

Search runs with GroupKFold on the training portion only — the held-out test
patients are never touched here — and writes reports/best_params.json, which
`pipeline.build_models` picks up automatically on the next training run.
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from .config import RANDOM_STATE, REPORT_DIR, TEST_SIZE
from .data_prep import build_dataset, split_feature_types
from .pipeline import assemble, build_models
from .train import grouped_split

warnings.filterwarnings("ignore")

N_ITER = 30
CV_FOLDS = 3  # 3 folds keeps the search under a couple of minutes; 5 for the final report

SEARCH_SPACES = {
    "xgboost": {
        "model__n_estimators": randint(200, 900),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(3, 9),
        "model__min_child_weight": randint(1, 30),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.5, 20),
        "model__gamma": uniform(0, 2),
    },
    "lightgbm": {
        "model__n_estimators": randint(200, 900),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__num_leaves": randint(15, 90),
        "model__min_child_samples": randint(10, 120),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.5, 20),
    },
}


def main() -> dict:
    X, y, groups = build_dataset()
    numeric, categorical = split_feature_types(X)
    X_full, _, y_full, _, g_full, _ = grouped_split(X, y, groups, TEST_SIZE)

    specs = build_models(y_full.to_numpy())
    results = {}

    for name, space in SEARCH_SPACES.items():
        t0 = time.time()
        search = RandomizedSearchCV(
            estimator=assemble(specs[name], numeric, categorical),
            param_distributions=space,
            n_iter=N_ITER,
            # PR-AUC, not accuracy: with an 11% positive rate, accuracy is
            # maximised by a model that never flags anyone.
            scoring="average_precision",
            cv=GroupKFold(n_splits=CV_FOLDS),
            random_state=RANDOM_STATE,
            n_jobs=1,  # the estimators already use every core
            refit=False,
            verbose=1,
        )
        search.fit(X_full, y_full, groups=g_full)

        # Round-tripping through JSON turns everything into a float; XGBoost and
        # LightGBM reject a float where they expect a count, so cast explicitly.
        integer_params = {
            "n_estimators", "max_depth", "min_child_weight", "num_leaves", "min_child_samples",
        }
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        best = {k: (int(v) if k in integer_params else float(v)) for k, v in best.items()}
        results[name] = best
        print(
            f"[tune] {name}: CV PR-AUC {search.best_score_:.4f} "
            f"(default rank baseline) in {time.time() - t0:.0f}s\n       {best}"
        )

    (REPORT_DIR / "best_params.json").write_text(json.dumps(results, indent=2))
    print(f"[tune] wrote {REPORT_DIR / 'best_params.json'}")
    return results


if __name__ == "__main__":
    main()
