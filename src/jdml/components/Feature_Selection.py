import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression

from ..Logger import logging

METHODS = ("none", "variance", "correlation", "importance", "mutual_info", "rfe")
CUMULATIVE_SHARE = 0.95


def as_numeric_frame(X):
    """
    A numeric copy of X for ranking columns: numbers are median imputed, dates become day counts
    and text columns become category codes.
    """
    out = pd.DataFrame(index=X.index)
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = (series - series.min()).dt.days.fillna(0)
        elif pd.api.types.is_numeric_dtype(series):
            series = series.astype(float)
            out[col] = series.fillna(series.median() if series.notna().any() else 0.0)
        else:
            out[col] = pd.factorize(series.astype(str))[0]
    return out


def unusable_columns(X):
    """
    Columns that cannot help a model: constant columns, and text or whole-number columns where
    every row has its own value (IDs, names).
    """
    dropped = {}
    n_rows = len(X)
    for col in X.columns:
        series = X[col]
        unique = series.nunique(dropna=True)
        if unique <= 1:
            dropped[col] = "constant"
        elif n_rows > 20 and not pd.api.types.is_float_dtype(series) \
                and not pd.api.types.is_datetime64_any_dtype(series) and unique == n_rows:
            dropped[col] = "looks like an ID (every value is unique)"
    return dropped


def _pick(scores, top_k):
    """Names to keep from a score Series: the top_k best, or enough to cover 95% of the total score."""
    ordered = scores.sort_values(ascending=False)
    if top_k:
        return list(ordered.index[:top_k])
    positive = ordered[ordered > 0]
    if positive.empty:
        return list(ordered.index[:1])
    covered = positive.cumsum() / positive.sum()
    return list(positive.index[: int((covered < CUMULATIVE_SHARE).sum()) + 1])


def _forest(problem, n_estimators, random_state):
    cls = RandomForestClassifier if problem == "Classification" else RandomForestRegressor
    return cls(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)


def _step(method, numeric, y, problem, top_k, threshold, random_state, X_raw):
    """Run one selection step on the numeric frame. Returns (kept columns, score per column)."""
    columns = list(numeric.columns)
    if method == "variance":
        top_share = X_raw.apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
        scores = 1 - top_share
        return [c for c in columns if top_share[c] < 0.99], scores
    if method == "correlation":
        corr = numeric.corr().abs().fillna(0)
        is_numeric = [pd.api.types.is_numeric_dtype(X_raw[c]) for c in columns]
        kept = []
        for col, numeric_col in zip(columns, is_numeric):
            if numeric_col and any(corr.loc[col, k] > threshold for k in kept if pd.api.types.is_numeric_dtype(X_raw[k])):
                continue
            kept.append(col)
        scores = pd.Series({c: corr.loc[c].drop(c).max() if len(columns) > 1 else 0.0 for c in columns})
        return kept, scores
    if method == "importance":
        forest = _forest(problem, 100, random_state).fit(numeric, y)
        scores = pd.Series(forest.feature_importances_, index=columns)
        return _pick(scores, top_k), scores
    if method == "mutual_info":
        discrete = [not pd.api.types.is_numeric_dtype(X_raw[c]) for c in columns]
        func = mutual_info_classif if problem == "Classification" else mutual_info_regression
        scores = pd.Series(func(numeric, y, discrete_features=discrete, random_state=random_state), index=columns)
        return _pick(scores, top_k), scores
    if method == "rfe":
        wanted = top_k or max(1, len(columns) // 2)
        rfe = RFE(_forest(problem, 50, random_state), n_features_to_select=min(wanted, len(columns))).fit(numeric, y)
        scores = pd.Series(1.0 / rfe.ranking_, index=columns)
        return [c for c, keep in zip(columns, rfe.support_) if keep], scores
    raise ValueError(f"Unknown feature selection method '{method}'; use one of {list(METHODS)}.")


def select_features(X, y, problem, methods, top_k=0, correlation_threshold=0.95, random_state=14, auto=False):
    """
    Choose the feature columns to train on, using the training rows only.

    X: DataFrame of candidate feature columns. y: numeric target (class codes for classification).
    methods: steps run in order (see METHODS); with auto=True constant and ID-like columns are
    dropped first. Returns (selected column list, report DataFrame with one row per candidate).
    """
    reasons = {}
    if auto or methods:
        reasons.update(unusable_columns(X))
    alive = [c for c in X.columns if c not in reasons]
    last_score = pd.Series(np.nan, index=X.columns)
    score_from = pd.Series("", index=X.columns, dtype=object)
    for method in methods:
        if method == "none" or len(alive) <= 1:
            continue
        numeric = as_numeric_frame(X[alive])
        kept, scores = _step(method, numeric, y, problem, top_k, correlation_threshold, random_state, X[alive])
        if not kept:
            kept = [scores.sort_values(ascending=False).index[0]]
        last_score.loc[scores.index] = scores
        score_from.loc[scores.index] = method
        for col in alive:
            if col not in kept:
                reasons[col] = f"dropped by {method}"
        alive = [c for c in alive if c in kept]
    if not alive:
        raise ValueError("Feature selection removed every column; list the features explicitly or relax the selection.")

    report = pd.DataFrame({
        "feature": list(X.columns),
        "score": [last_score[c] for c in X.columns],
        "score_from": [score_from[c] for c in X.columns],
        "kept": [c in alive for c in X.columns],
        "reason": [reasons.get(c, "") for c in X.columns],
    }).sort_values(["kept", "score"], ascending=[False, False], na_position="last")
    logging.info(f"Feature selection kept {len(alive)} of {X.shape[1]} columns: {alive}")
    return alive, report.reset_index(drop=True)
