"""
Illustrative feature-engineering sketch for the House Prices - Advanced Regression Techniques competition.

This file mirrors the structure of the private solution (cleaning -> engineered columns -> categorical handling -> ordinal maps), 
but keeps the details intentionally general. 

Hyperparameters, rare-category rules, target-encoding columns, cluster settings, and the full feature set from the
competition submission are omitted on purpose.
"""

from __future__ import annotations

import pandas as pd 

QUALITY_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    The competition version treats "missing" differently by feature family
    (e.g., no garage / no basement -> absence labels / zeros)
    Exact column lists and strategies are kept private. 
    """
    data = df.copy()

    # Categorical Absence can be meaningful, not just Null
    for col in ["Alley", "Fence", "Fireplace"]:
        if col in data.columns:
            data[col] = data[col.fillna("None")]

    # Numeric size fields often mean "feature not present"
    for col in ["MasVnrArea", "GarageArea", "TotalBsmtSF"]:
        if col in data.columns:
            data[col] = data[col.fillna(0)]

    # Remaining numerics can be a simple median fill for this example only
    numeric_cols = data.select_dtypes(include="number").columns
    for col in numeric_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    return data

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """        
    Creates additional housing features. 

    The private pipeline adds many more aggregates, ratios, seasonality 
    encodings and interaction terms. Those are omitted here. 
    """
    data = df.copy()

    # Size aggregates (illustrative)
    data["TotalSF"] = (
        data.get("TotalBsmtSF", 0)
        + data.get("1stFlrSF", 0)
        + data.get("2ndFlrSF", 0)
    )

    if "GrLivArea" in data.columns and "TotalBsmtSF" in data.columns:
        data["LivingPlusBsmt"] = data["GrLivArea"] + data["TotalBsmtSF"]
    
    # Age / remodal signals

    if {"YrSold", "YearBuilt"}.issubset(data.columns):
        data["Age"] = data["YrSold"] - data["YearBuilt"] 
        
    if {"YrSold", "YearRemodAdd"}.issubset(data.columns):    
        data["IsRemodeled"] = (data["YearBuilt"] != data["YearRemodAdd"]).astype(int)

    # Bathroom
    if {"FullBath", "HalfBath"}.issubset(data.columns):
        data["TotalBath"] = data["FullBath"] + 0.5 * data["HalfBath"]
        # Basement bath terms used in the private version are not shown.
    
    # Quality interactions 
    if {"OverallQual", "GrLivArea"}.issubset(data.columns):
        data["Qual_x_SF"] = data["OverallQual"] * data["GrLivArea"]

    if "OverallQual" in data.columns:
        data["OverallQual_sq"] = data["OverallQual"] ** 2

    return data 

def encode_ordinal_quality(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    """
    Map a few ordered quality strings to integers.

    Private solution covers a larger ORDINAL_MAPS disctionary across
    exterior, basement, kitchen, garage, fence etc. 
    """
    for col in ["ExterQual", "KitchenQual", "HeatingOC"]:
        if col in data.columns:
            data[col] = data[col].map(QUALITY_MAP).fillna(0)

    return data 


def bucket_rare_categories_example(
        df: pd.DataFrame,
        cols: list[str] | None = None,
        min_count: int = 10
) -> pd.DataFrame:
    
    """
    Collapse infrequent category levels into a shared "Rare" label.

    Final competition min_count / column list omitted
    Determinded through private CV experiments
    """
    data = df.copy()
    cols = cols or [c for c in data.columns if data[c].dtype == object]

    for col in cols:
        counts = data[col].value_counts()
        rare = counts[counts < min_count].index
        data[col] = data[col].where(data[col].isin(rare), other="Rare")

    return data 

def target_encode_example(
        train: pd.DataFrame,
        test: pd.DataFrame,
        col: str,
        target: str = "SalePrice",
        smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Smoothed mean encoding sketch for a single high-cardinally column.

    Private solution:
    - encodes multiple columns
    - fits encodings on training folds only (no leakage)
    - uses a tuned smoothing factor

    Final competition TARGET_ENCODE_COLS / smoothing omitted.
    """

    global_mean = train[target].mean()
    stats = train.groupby(col)[target].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )

    train = train.copy()
    test = test.copy()
    train[f"(col)_te"] = train[col].map(smooth)
    test[f"(col)_te"] = test[col].map(smooth).fillna(global_mean)
    
    return train, test 

def preprocess_example(df: pd.DataFrame) -> pd.DataFrame:

    """
    End-to-end example preprocess pass. 

    Order of operations matches the private pipeline at a high level
    fill missing -> engineer features -> ordinal maps -> rare categories
    
    Steps left out of this public repo:
    - additional feature transformations
    - additional model-specific preprocessing
    - final optimisation choices
    """

    data = fill_missing_values(df)
    data = create_features(data)
    data = encode_ordinal_quality(data)
    data = bucket_rare_categories_example(data)

    # Id is unused as a model feature in the real pipeline.

    return data.drop(columns="Id", errors="ignore")

if __name__ == "__main__":
    # Demo stub. Replace with paths to Kaggle train.csv if exploring locally. 
    print(
        "example_preprocess.py is a portfolio sketch only.\n"
        "The competition-winning preprocess.py remains private"
    )