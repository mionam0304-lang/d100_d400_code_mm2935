import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from modeling import load_split_xy, make_preprocessor, rmse

X_train, y_train, X_test, y_test = load_split_xy("data/jobs_cleaned.parquet")

pre = make_preprocessor()

# ----------------------------
# GLM baseline (ElasticNet on log-target)
# ----------------------------
glm = ElasticNet(alpha=1e-2, l1_ratio=0.5, max_iter=5000, random_state=42)
glm_pipe = Pipeline([("preprocess", pre), ("model", glm)])

y_train_log = np.log1p(y_train)
glm_pipe.fit(X_train, y_train_log)

pred_glm_log = glm_pipe.predict(X_test)
pred_glm = np.expm1(pred_glm_log)

# print("=== GLM baseline (ElasticNet, log-target) ===")
print("MAE :", mean_absolute_error(y_test, pred_glm))
print("RMSE:", rmse(y_test, pred_glm))


# ----------------------------
# LGBM baseline (log-target)
# ----------------------------
lgbm = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1,
)

lgbm_pipe = Pipeline([("preprocess", pre), ("model", lgbm)])

lgbm_pipe.fit(X_train, y_train_log)

pred_lgbm_log = lgbm_pipe.predict(X_test)
pred_lgbm = np.expm1(pred_lgbm_log)

# print("\n=== LGBM baseline (log-target) ===")
print("MAE :", mean_absolute_error(y_test, pred_lgbm))
print("RMSE:", rmse(y_test, pred_lgbm))
