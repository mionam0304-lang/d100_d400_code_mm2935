import numpy as np
import polars as pl
from joblib import load

from data import create_sample_split
from evaluating import evaluate_predictions
from modeling._common import CAT_COLS, NUM_COLS, TARGET

# ----------------------------
# Load cleaned data + split (same deterministic split)
# ----------------------------
df = pl.read_parquet("data/jobs_cleaned.parquet")
df = create_sample_split(df, id_column="job_id", training_frac=0.8)

test_df = df.filter(pl.col("sample") == "test")

feature_cols = NUM_COLS + CAT_COLS

# scikit-learn pipelines want panmdas-like inputs
X_test = test_df.select(feature_cols).to_pandas()
y_test = test_df.select(TARGET).to_numpy().ravel()

# ----------------------------
# Load tuned models
# ----------------------------
best_glm = load("modeling/models/best_glm.joblib")
best_lgbm = load("modeling/models/best_lgbm.joblib")

# ----------------------------
# Predict
# (GLM is ElasticNet on log-target)
# ----------------------------
pred_glm = np.expm1(best_glm.predict(X_test))

# (LGBM is trained on log-target)
pred_lgbm = np.expm1(best_lgbm.predict(X_test))

# ----------------------------
# Evaluate using PS4-style function
# ----------------------------
eval_df = X_test.copy()
eval_df[TARGET] = y_test
eval_df["pred_glm"] = pred_glm
eval_df["pred_lgbm"] = pred_lgbm

print("=== Evaluation: Tuned GLM (ElasticNet, log-target) ===")
print(evaluate_predictions(eval_df, TARGET, preds_column="pred_glm", tweedie_power=1.5))

print("\n=== Evaluation: Tuned LGBM (log-target) ===")
print(
    evaluate_predictions(eval_df, TARGET, preds_column="pred_lgbm", tweedie_power=1.5)
)
