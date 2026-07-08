# Kaggle House Prices - Advanced Regression Techniques

End-to-end machine learning pipeline for predicting residential sale prices in Ames, Iowa using advanced feature engineering, per-model hyperparameter optimisation and weighted ensemble learning.

This project achieved a **Top 200 worldwide Kaggle ranking** in the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition with a **Leaderboard RMSE** of 0.11793.

![Leaderboard Result](images/leaderboard.png)

> **Note on code availability:** The final competition submission pipeline is kept private, since publishing it would allow others to reproduce the exact solution for an active competition where this result ranks in the top 200. This repository contains simplified example implementations demonstrating the architecture, preprocessing workflow, and modelling approach without exposing the final submission code.

---

# Project Overview

The objective of this competition is to predict residential house sale prices from 79 property features, including:
- Overall quality and condition
- Living area and room sizes
- Year built and renovation history
- Basement and garage characteristics
- Neighborhood information
- External features and amenities

## Evaluation Metric

Submissions are scored using Root Mean Squared Error (RMSE) between the logarithm of the predicted price and the logarithm of the observed sale price - meaning proportional errors on expensive and cheap houses are penalised equally, which shaped several of the preprocessing decisions below.

$$
\text{RMSE} =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left(
\log(\hat{y}_i)-\log(y_i)
\right)^2
}
$$

---

# Pipeline Architecture

![Machine Learning Pipeline](images/pipeline.png)

Both `train.csv` and `test.csv` pass through the same cleaning, feature engineering, and preprocessing steps, fitted on the training data and applied consistently to the test set to avoid leakage.

Each of the five models is tuned and validated independently - Optuna searches a model-specific hyperparameter space, evaluated using k-fold cross-validation, rather than sharing a single tuning pass across all models. Final predictions are generated independently from each model and combined using an ensemble strategy.

---

# Feature Engineering

Rather than list every transformation, the notes below cover the reasoning behind the less obvious choices:

- **Ordinal encoding over one-hot for quality features.** Fields like `ExterQual` or `KitchenQual` have a natural rank. One-hot encoding would discard that ordering and force models to relearn it from scratch; ordinal encoding preserves it directly.
- **Missing values handled by meaning, not blanket imputation.** Many "missing" values in this dataset mean the feature doesn't apply (e.g. no garage, no basement) rather than data being genuinely absent. These were encoded as an explicit "none" category instead of median/mean imputation, which would have implied a typical garage existed where none does.
- **Derived area and age features.** Total living area, total bathroom count, and property/remodel age were engineered because tree-based models split on individual columns - they do not automatically combine related measurements into higher-level domain features.
- **Aggregated quality indicators.** Combining related quality/condition scores into a single index reduced redundant, highly correlated columns competing for importance during tuning.
- **Interaction features** were added selectively where domain knowledge suggested an effect that's multiplicative rather than additive (e.g. quality mattering more in larger homes).

---

# Models Used

The final ensemble combines five complementary models, each independently tuned:

| Model | Purpose |
|---|---|
| LightGBM | Strong tabular baseline with efficient gradient boosting |
| XGBoost | Alternative boosting approach capturing different tree structures |
| CatBoost | Handles categorical relationships effectively |
| HistGradientBoosting | Lightweight sklearn boosting model |
| Ridge Regression | Adds linear diversity to reduce correlated errors |

**Why an ensemble, and why these five:** the four boosting models capture non-linear feature interactions well but tend to make correlated errors, since they're built on similar splitting logic. Ridge Regression was included specifically for its *different* error profile - as a linear model, it fits the parts of the target that are well-explained by simple linear combinations, which boosting models can occasionally overfit around. Even where Ridge's standalone CV score was the weakest of the five, it still improved the blended result, because ensembling benefits more from prediction *diversity* than from any single model's raw accuracy.

Ensemble weights were set based on out-of-fold cross-validation performance for each model, rather than fixed evenly or hand-tuned to the public leaderboard - this reduces the risk of the weighting overfitting to leaderboard noise rather than genuine predictive strength.

---

# Optimisation & Engineering

Key engineering decisions used to improve model reliability and generalisation:

- **Cross-validation strategy:** Used k-fold cross-validation with out-of-fold predictions to evaluate models and support ensemble optimisation.
- **Model-specific optimisation:** Each model was tuned independently using appropriate hyperparameter search spaces rather than applying a single configuration across different model families.
- **Reusable preprocessing pipeline:** Feature preparation was implemented as a consistent transformation workflow applied identically during training and inference, reducing the risk of train/test inconsistencies.
- **Reproducibility:** Random seeds were controlled across experiments to ensure results could be reproduced reliably.

---

# Technologies

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Optuna
- Joblib

---
# Installation + Setup Demo

Install dependencies for the public example implementation:

```bash
pip install -r requirements-example.txt
```

```bash
python example_model.py
```
---

# License

The example implementation and documentation are provided under the MIT License.

The final competition submission pipeline, including the exact preprocessing, feature engineering, optimisation configuration, and ensemble strategy used for the leaderboard result, is not included.