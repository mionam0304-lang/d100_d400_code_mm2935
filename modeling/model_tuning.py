import numpy as np
from joblib import dump
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from modeling import load_split_xy, make_preprocessor, rmse

X_train, y_train, X_test, y_test = load_split_xy("data/jobs_cleaned.parquet")

pre = make_preprocessor()
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scorer = make_scorer(rmse, greater_is_better=False)

y_train_log = np.log1p(y_train)

# ----------------------------
# (1) GLM tuning: alpha + l1_ratio
# ----------------------------
glm_pipe = Pipeline(
    [
        ("preprocess", pre),
        ("model", ElasticNet(max_iter=5000, random_state=42)),
    ]
)

glm_param_dist = {
    "model__alpha": loguniform(1e-4, 1e1),
    "model__l1_ratio": np.linspace(0.0, 1.0, 6),
}

glm_search = RandomizedSearchCV(
    glm_pipe,
    glm_param_dist,
    n_iter=30,
    scoring=rmse_scorer,
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

glm_search.fit(X_train, y_train_log)
best_glm = glm_search.best_estimator_

pred_glm = np.expm1(best_glm.predict(X_test))

# print("\n=== Tuned GLM (ElasticNet, log-target) ===")
print("Best params:", glm_search.best_params_)
print("MAE :", mean_absolute_error(y_test, pred_glm))
print("RMSE:", rmse(y_test, pred_glm))


# ----------------------------
# (2) LGBM tuning: learning_rate, n_estimators, num_leaves, min_child_weight
# ----------------------------
lgbm_pipe = Pipeline(
    [
        ("preprocess", pre),
        ("model", LGBMRegressor(random_state=42)),
    ]
)

lgbm_param_dist = {
    "model__learning_rate": loguniform(1e-3, 2e-1),
    "model__num_leaves": [15, 31, 63, 127],
    "model__min_child_weight": loguniform(1e-3, 1e1),
    "model__n_estimators": [300, 600, 1000, 2000],
}

lgbm_search = RandomizedSearchCV(
    lgbm_pipe,
    lgbm_param_dist,
    n_iter=30,
    scoring=rmse_scorer,
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

lgbm_search.fit(X_train, y_train_log)
best_lgbm = lgbm_search.best_estimator_

pred_lgbm = np.expm1(best_lgbm.predict(X_test))

# print("\n=== Tuned LGBM (log-target) ===")
print("Best params:", lgbm_search.best_params_)
print("MAE :", mean_absolute_error(y_test, pred_lgbm))
print("RMSE:", rmse(y_test, pred_lgbm))

dump(best_glm, "modeling/models/best_glm.joblib")
dump(best_lgbm, "modeling/models/best_lgbm.joblib")

print("Best CV score (GLM):", glm_search.best_score_)
print("Best CV score (LGBM):", lgbm_search.best_score_)
