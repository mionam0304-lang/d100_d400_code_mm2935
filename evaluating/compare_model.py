from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from joblib import load
from sklearn.inspection import PartialDependenceDisplay

from data import create_sample_split
from evaluating import evaluate_predictions
from modeling._common import CAT_COLS, NUM_COLS, TARGET

# ----------------------------
# Settings
# ----------------------------
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDP_DIR = OUT_DIR / "pdp"
PDP_DIR.mkdir(parents=True, exist_ok=True)


def predicted_vs_actual(y_true, y_pred, title: str, out_path: Path) -> None:
    # Predicted vs Actual plot
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.3)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual salary_usd")
    plt.ylabel("Predicted salary_usd")
    plt.title(title)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------
# Load cleaned data + split
# ----------------------------
df = pl.read_parquet("data/jobs_cleaned.parquet")
df = create_sample_split(df, id_column="job_id", training_frac=0.8)
test_df = df.filter(pl.col("sample") == "test")

feature_cols = NUM_COLS + CAT_COLS
X_test = test_df.select(feature_cols).to_pandas()
y_test = test_df.select(TARGET).to_numpy().ravel()

# ----------------------------
# Load tuned models
# ----------------------------
best_glm = load("modeling/models/best_glm.joblib")  # Pipeline(preprocess + ElasticNet)
best_lgbm = load("modeling/models/best_lgbm.joblib")  # Pipeline(preprocess + LGBM)

# ----------------------------
# Predict (both trained on log-target)
# ----------------------------
pred_glm = np.expm1(best_glm.predict(X_test))
pred_lgbm = np.expm1(best_lgbm.predict(X_test))

# ----------------------------
# PS4-style evaluation table
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

# ----------------------------
# Predicted vs Actual plots
# ----------------------------
predicted_vs_actual(
    y_test,
    pred_glm,
    "Predicted vs Actual (Tuned GLM)",
    OUT_DIR / "pred_vs_actual_glm.png",
)

predicted_vs_actual(
    y_test,
    pred_lgbm,
    "Predicted vs Actual (Tuned LGBM)",
    OUT_DIR / "pred_vs_actual_lgbm.png",
)

# ----------------------------
# Feature importance (Top 5) for LGBM
# ----------------------------
pre = best_lgbm.named_steps["preprocess"]
model = best_lgbm.named_steps["model"]

feature_names = pre.get_feature_names_out()
importances = model.feature_importances_

top5_idx = np.argsort(importances)[::-1][:5]
top5_names = [feature_names[i] for i in top5_idx]

print("\nTop 5 features (LGBM gain importance):")
for i, name in enumerate(top5_names, 1):
    print(f"{i}. {name}")

# Bar plot / 棒グラフ
plt.figure()
plt.bar(range(1, 6), [importances[i] for i in top5_idx])
plt.xticks(range(1, 6), top5_names, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance (gain)")
plt.title("Top 5 Feature Importances (LGBM)")
plt.savefig(OUT_DIR / "lgbm_top5_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------
# Partial Dependence Plots (PDP) for top 5
# ----------------------------
# Compute PDP on preprocessed matrix to avoid name issues

X_test_pre = pre.transform(X_test)

if hasattr(X_test_pre, "toarray"):
    X_test_pre = X_test_pre.toarray()

for idx, fname in zip(top5_idx, top5_names):
    plt.figure()
    PartialDependenceDisplay.from_estimator(
        model,
        X_test_pre,
        features=[idx],
        feature_names=feature_names,
        kind="average",
    )
    plt.title(f"PDP (LGBM): {fname}")
    safe_name = fname.replace(":", "_").replace(" ", "_").replace("/", "_")
    plt.savefig(PDP_DIR / f"pdp_{safe_name}.png", dpi=200, bbox_inches="tight")
    plt.close()
