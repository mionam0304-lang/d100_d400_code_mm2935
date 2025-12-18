import numpy as np
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import auc


def lorenz_curve(y_true, y_pred, exposure):
    """Lorenz curve for ranking performance (PS4 reference)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_outcome = y_true[ranking]
    cumulated_outcome = np.cumsum(ranked_outcome * ranked_exposure)
    cumulated_outcome /= cumulated_outcome[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_outcome))
    return cumulated_samples, cumulated_outcome


def evaluate_predictions(
    df,
    outcome_column,
    *,
    preds_column=None,
    model=None,
    tweedie_power=1.5,
    exposure_column=None,
):
    """Evaluate predictions against actual outcomes (PS4 reference style)."""

    evals = {}

    assert preds_column or model, "Please either provide the column name of"
    " the pre-computed predictions or a model to predict from."
    if preds_column is None:
        preds = model.predict(df)
    else:
        preds = df[preds_column]

    if exposure_column:
        weights = df[exposure_column]
    else:
        weights = np.ones(len(df))

    evals["mean_preds"] = np.average(preds, weights=weights)
    evals["mean_outcome"] = np.average(df[outcome_column], weights=weights)
    evals["bias"] = (evals["mean_preds"] - evals["mean_outcome"]) / evals[
        "mean_outcome"
    ]

    evals["mse"] = np.average((preds - df[outcome_column]) ** 2, weights=weights)
    evals["rmse"] = np.sqrt(evals["mse"])
    evals["mae"] = np.average(np.abs(preds - df[outcome_column]), weights=weights)

    evals["deviance"] = TweedieDistribution(tweedie_power).deviance(
        df[outcome_column], preds, sample_weight=weights
    ) / np.sum(weights)

    ordered_samples, cum_actuals = lorenz_curve(df[outcome_column], preds, weights)
    evals["gini"] = 1 - 2 * auc(ordered_samples, cum_actuals)

    return pd.DataFrame(evals, index=[0]).T
