"""
Illustrative modeling sketch for the House Prices - Advanced Regression Techniques competition.

Shows the same architecture as the private solution:
  - train on log(SalePrice) (matches the competition metric)
  - multiple heterogeneous regressors
  - K-fold out-of-fold predictions
  - blend / stack style ensembling

  Tuned hyperparameters, early-stopping schedules, CV fold count used in the final submission, blend weights, and stacking details are omitted.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional boosters - private solution uses LightGBM / XGBoost / CAtBoost
# with tuned settings. Exact hyperparams omitted.

try: 
    from lightgbm import LGBMRegressor

    _HAS_LGBM = True
except Exception: # no cover - optional dependency
    _HAS_LGBM = False

RANDOM_STATE = 42
# Final competition N_CV_FOLDS omitted
# Determinded through private CV experiments.
N_CV_FOLDS = 5

def log_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Competition-style metric: RMSE between log(pred) and log(actual).
    y_pred = np.maximum(y_pred, 1.0)
    return float(np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred))))

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Private solution uses model-specific preprocessors (e.g. native categorical handling for CatBoost, scaled features for Ridge).
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
                    (
                        "num",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        numeric,
                    ),
                    (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )

def build_base_models() -> dict[str, object]:
    """
    Placeholder base learners for the portfolio sketch.
    
    Final competition model list / hyperparameters omitted.
    Determined through private CV experiments.
    
    Private ensemble included gradient boosting variants + a linear model, each trained on log targets with early stopping where applicable.
    """
    models: dict[str, object] = {
        "hist_gbr": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=RANDOM_STATE,
        ),
        "ridge": Ridge(alpha=10.0),
    }

    if _HAS_LGBM:
            models["lightgbm"] = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=RANDOM_STATE,
                verbose=-1,
            )

    return models

def fit_one_model(
    model: object,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[ColumnTransformer, object]:
    # Fit preprocessor + model on log(SalePrice)
    X_t = preprocessor.fit_transform(X_train)
    y_log = np.log(y_train.to_numpy())
    model.fit(X_t, y_log)
    return preprocessor, model

def predict_one_model(
    preprocessor: ColumnTransformer,
    model: object,
    X: pd.DataFrame,
) -> np.ndarray:
    # Invert the log transform to return prices on the original scale.
    X_t = preprocessor.transform(X)
    return np.exp(model.predict(X_t))

def optimise_blend_weights_example(
    oof_matrix: np.ndarray,
    y: pd.Series,
) -> np.ndarray:
    """
    Example non-negative blending weights that sum to 1.
    
    Private solution optimises weights (or compares against a log-space stacking meta-model) using out-of-fold predictions and the competition
    log-RMSE objective.
    
    Final competition blend weights omitted.
    Determined through private CV experiments.
    """
    # Portfolio placeholder: equal weights.
    # Do not use these for a competitive submission.
    n_models = oof_matrix.shape[1]
    return np.ones(n_models) / n_models

def cross_validated_ensemble_example(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, dict[str, float]]:
    """
    Sketch of the private CV loop:
    1. For each fold, fit each base model on the train split
    2. Collect out-of-fold predictions
    3. Blend OOF predictions and report log-RMSE
    
    Early stopping, parallel model fits within a fold, median best-iteration refits on full data, and stack-vs-blend selection are private.
    """
    models = build_base_models()
    model_names = list(models.keys())
    oof = {name: np.zeros(len(y)) for name in model_names}

    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in cv.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        for name, model in models.items():
            # Fresh preprocessor + fresh model instance each fold
            pre = build_preprocessor(X_tr)
            fresh = build_base_models()[name]
            pre, fitted = fit_one_model(fresh, pre, X_tr, y_tr)
            oof[name][val_idx] = predict_one_model(pre, fitted, X_va)
    oof_matrix = np.column_stack([oof[name] for name in model_names])
    weights = optimise_blend_weights_example(oof_matrix, y)
    blended = oof_matrix @ weights
    score = log_rmse(y.to_numpy(), blended)
    
    weight_map = {name: float(w) for name, w in zip(model_names, weights)}
    return score, weight_map

def main() -> None:
    """
    Portfolio entrypoint.
    
    Not wired to train.csv / submission.csv on purpose - this repo is a public write-up of the approach, not a runnable reproduction of the
    private Kaggle solution.
    """
    print("example_model.py is a portfolio sketch only.")
    print("The competition-winning model.py remains private.")
    print()
    print("Approach summary:")
    print("  - Predict log(SalePrice), then exp() for submission scale")
    print("  - Ensemble heterogeneous regressors via out-of-fold blending")
    print("  - Select blend vs stack using private CV (details omitted)")
    print()
    print(f"Example CV folds shown here: {N_CV_FOLDS} (not the private setting)")
    print(f"Example base models: {list(build_base_models().keys())}")

if __name__ == "__main__":
    main()