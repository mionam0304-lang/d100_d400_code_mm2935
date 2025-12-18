from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import create_sample_split

# ----------------------------
# Columns
# ----------------------------
TARGET = "salary_usd"

NUM_COLS = [
    "years_experience",
]

CAT_COLS = [
    "employment_type",
    "company_location",
    "industry_group",
    "education_required",
    "company_size",
]


def rmse(y_true, y_pred) -> float:  # RMSE = sqrt(MSE)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_preprocessor() -> ColumnTransformer:
    # Feature engineering with sklearn transformers
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
    )


def load_split_xy(
    parquet_path: str = "data/jobs_cleaned.parquet",
    id_column: str = "job_id",
    training_frac: float = 0.8,
):
    """
    Load cleaned data, create deterministic ID-based split, and return X/y.
    """
    df = pl.read_parquet(parquet_path)
    df = create_sample_split(df, id_column=id_column, training_frac=training_frac)

    train_df = df.filter(pl.col("sample") == "train")
    test_df = df.filter(pl.col("sample") == "test")

    feature_cols = NUM_COLS + CAT_COLS

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select(TARGET).to_numpy().ravel()

    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select(TARGET).to_numpy().ravel()

    return X_train, y_train, X_test, y_test
