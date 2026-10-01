"""Predeclared candidate comparison using Batch 1 training-policy groups only."""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
CORE = ["delta_logvar", "QD_slope_10_100", "mean_chargetime", "mean_Tavg"]
POLICY = [*CORE, "C1", "switch_SOC", "C2"]
EXPANDED = [*POLICY, "std_QD", "mean_IR", "IR_change"]
FEATURE_SETS = {
    "delta": ["delta_logvar"], "delta_qd": CORE[:2], "core": CORE,
    "policy": POLICY, "expanded": EXPANDED,
    "duplicates": [*CORE, "delta_min", "delta_mean", "delta_range", "mean_Tmax"],
    "core_category": [*CORE, "policy"], "expanded_category": [*EXPANDED, "policy"],
}


class CorrelationPruner(TransformerMixin, BaseEstimator):
    """Fit on the current training fold; preserve predeclared feature priority."""
    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        self.n_features_in_ = a.shape[1]
        self.feature_names_in_ = np.asarray(getattr(X, "columns", [f"x{i}" for i in range(a.shape[1])]), dtype=object)
        keep, removed = [], []
        for i in range(a.shape[1]):
            if np.std(a[:, i]) < 1e-12:
                removed.append((i, "constant", None))
                continue
            redundant = None
            for j in keep:
                if abs(np.corrcoef(a[:, i], a[:, j])[0, 1]) >= self.threshold:
                    redundant = j
                    break
            if redundant is None:
                keep.append(i)
            else:
                removed.append((i, "high_correlation", redundant))
        if not keep:
            keep = [0]
        self.keep_indices_, self.removed_ = keep, removed
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)
        return names[self.keep_indices_]


def build_model(spec):
    features = FEATURE_SETS[spec["feature_set"]]
    numeric = [f for f in features if f != "policy"]
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("prune", CorrelationPruner(spec["corr_threshold"]))]
    if spec["family"] in ("Ridge", "ElasticNet"):
        numeric_steps.append(("scale", StandardScaler()))
    transforms = [("num", Pipeline(numeric_steps), numeric)]
    if "policy" in features:
        cat_steps = [("impute", SimpleImputer(strategy="constant", fill_value="missing"))]
        if spec["family"] != "CatBoost":
            cat_steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transforms.append(("cat", Pipeline(cat_steps), ["policy"]))
    preprocess = ColumnTransformer(transforms, remainder="drop").set_output(transform="pandas")
    params = spec["params"]
    if spec["family"] == "Median":
        regressor = DummyRegressor(strategy="median")
    elif spec["family"] == "Ridge":
        regressor = Ridge(**params)
    elif spec["family"] == "ElasticNet":
        regressor = ElasticNet(**params, max_iter=30000, tol=1e-6, random_state=SEED)
    elif spec["family"] in ("RandomForest", "ExtraTrees"):
        cls = RandomForestRegressor if spec["family"] == "RandomForest" else ExtraTreesRegressor
        regressor = cls(**params, n_estimators=200, random_state=SEED, n_jobs=1)
    else:
        regressor = CatBoostRegressor(**params, iterations=200, learning_rate=0.03,
                                     random_seed=SEED, verbose=False, allow_writing_files=False,
                                     thread_count=1, loss_function="RMSE", cat_features=["cat__policy"])
    pipeline = Pipeline([("preprocess", preprocess), ("model", regressor)])
    return (TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
            if spec["log_target"] else pipeline)


def candidate_specs():
    specs = []
    def add(family, sets, grid, logs=(False, True), thresholds=(0.85,)):
        for fs in sets:
            for params in ParameterGrid(grid):
                for log in logs:
                    for threshold in thresholds:
                        specs.append({"family": family, "feature_set": fs, "params": params,
                                      "log_target": log, "corr_threshold": threshold})
    add("Median", ["delta"], {}, logs=(False,))
    add("Ridge", ["delta", "delta_qd", "core", "policy", "expanded", "duplicates", "core_category"],
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]})
    add("Ridge", ["duplicates"], {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, thresholds=(1.01,))
    add("ElasticNet", ["core", "policy", "expanded"], {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.8]})
    for family in ("RandomForest", "ExtraTrees"):
        add(family, ["core", "policy", "expanded"], {"max_depth": [2, 4], "min_samples_leaf": [3, 6]})
    add("CatBoost", ["core_category", "expanded_category"], {"depth": [2, 4], "l2_leaf_reg": [3.0, 10.0]})
    for i, spec in enumerate(specs):
        spec["candidate_id"] = f"C{i:03d}"
    return specs


def split_manifest(root, features):
    b1 = features.loc[features.batch == "B1"].copy()
    plan = pd.read_csv(root / "day1/output/planned_b1_split.csv")
    assert set(plan.cell_id) == set(b1.cell_id), "Data population changed: explicitly revise split plan."
    merged = b1.merge(plan, on="cell_id", suffixes=("", "_planned"), validate="one_to_one")
    assert (merged.policy == merged.policy_planned).all()
    assert np.array_equal(merged.cycle_life, merged.cycle_life_planned)
    train = merged.loc[merged.planned_split == "Train for CV", features.columns].copy().reset_index(drop=True)
    valid = merged.loc[merged.planned_split == "Valid hold-out", features.columns].copy().reset_index(drop=True)
    test = features.loc[features.batch == "B2"].copy().reset_index(drop=True)
    assert not set(train.policy) & set(valid.policy)
    assert not set(train.cell_id) & set(valid.cell_id)
    folds = list(GroupKFold(n_splits=5).split(train, train.cycle_life, train.policy))
    fold_for_cell = np.full(len(train), -1)
    for k, (tr, va) in enumerate(folds):
        assert not set(train.iloc[tr].policy) & set(train.iloc[va].policy)
        fold_for_cell[va] = k + 1
    manifest = pd.concat([
        train[["cell_id", "batch", "policy", "cycle_life"]].assign(split="Train OOF", cv_fold=fold_for_cell),
        valid[["cell_id", "batch", "policy", "cycle_life"]].assign(split="Valid", cv_fold=pd.NA),
        test[["cell_id", "batch", "policy", "cycle_life"]].assign(split="Test", cv_fold=pd.NA)], ignore_index=True)
    manifest.to_csv(root / "day2/output/split_manifest.csv", index=False)
    return train, valid, test, folds, manifest


def compare_candidates(root, train, folds):
    """Neither hold-out nor Batch 2 is accepted by the search function."""
    specs = candidate_specs()
    out = root / "day2/output"
    (out / "search_space.json").write_text(json.dumps(specs, indent=2))
    rows, fold_rows = [], []
    previous = None
    with threadpool_limits(limits=1):
        for spec in specs:
            label = (spec["family"], spec["feature_set"], spec["corr_threshold"])
            if label != previous:
                print("CV:", *label, flush=True)
                previous = label
            scores, selected_counts, warning_count = [], [], 0
            X, y = train[FEATURE_SETS[spec["feature_set"]]], train.cycle_life
            for k, (tr, va) in enumerate(folds):
                model = build_model(spec)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X.iloc[tr], y.iloc[tr])
                warning_count += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                pred = model.predict(X.iloc[va])
                assert np.isfinite(pred).all()
                score = 100 * mean_absolute_percentage_error(y.iloc[va], pred)
                pipe = model.regressor_ if spec["log_target"] else model
                selected_counts.append(len(pipe.named_steps["preprocess"].get_feature_names_out()))
                scores.append(score)
                fold_rows.append({"candidate_id": spec["candidate_id"], "fold": k+1,
                                  "train_cells": len(tr), "valid_cells": len(va), "MAPE_pct": score,
                                  "transformed_features": selected_counts[-1]})
            rows.append({**{k: spec[k] for k in ("candidate_id", "family", "feature_set", "log_target", "corr_threshold")},
                         "params": json.dumps(spec["params"], sort_keys=True),
                         "CV_MAPE_pct": float(np.mean(scores)), "CV_std_pct": float(np.std(scores, ddof=1)),
                         "mean_transformed_features": float(np.mean(selected_counts)),
                         "convergence_warnings": warning_count})
    comparison = pd.DataFrame(rows).sort_values(["CV_MAPE_pct", "CV_std_pct", "candidate_id"]).reset_index(drop=True)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "cv_fold_results.csv", index=False)
    best = next(s for s in specs if s["candidate_id"] == comparison.iloc[0].candidate_id)
    # Freeze before either external evaluation; no downstream test-based model revision.
    (out / "selected_model_config.json").write_text(json.dumps(best, indent=2))
    family_best = comparison.groupby("family", sort=False).head(1)
    family_best.to_csv(out / "model_family_best.csv", index=False)
    comparison.groupby(["family", "feature_set", "corr_threshold"], sort=False).head(1).to_csv(out / "feature_ablation.csv", index=False)
    print("Selected by B1 CV:", best, flush=True)
    return best, comparison


def evaluate_final(root, train, valid, test, folds, spec, comparison):
    out = root / "day2/output"
    names = FEATURE_SETS[spec["feature_set"]]
    model = build_model(spec)
    model.fit(train[names], train.cycle_life)
    oof = np.empty(len(train))
    for tr, va in folds:
        fold_model = build_model(spec)
        fold_model.fit(train.iloc[tr][names], train.iloc[tr].cycle_life)
        oof[va] = fold_model.predict(train.iloc[va][names])
    predictions, metric_rows = [], []
    for label, data, pred in [("Train OOF", train, oof), ("Valid", valid, model.predict(valid[names])),
                              ("Test", test, model.predict(test[names]))]:
        y = data.cycle_life.to_numpy()
        assert np.isfinite(pred).all() and (y > 0).all()
        frame = data[["cell_id", "batch", "policy", "cycle_life"]].copy()
        frame["split"], frame["prediction"] = label, pred
        frame["residual"] = pred-y
        frame["absolute_error"] = np.abs(pred-y)
        frame["APE_pct"] = 100*np.abs(pred-y)/y
        frame["life_group"] = np.select([y < 500, y > 1000], ["short <500", "long >1000"], default="middle 500-1000")
        frame["unseen_policy"] = ~frame.policy.isin(set(train.policy))
        frame["new_structure"] = frame.policy.str.contains("newstructure", regex=False)
        predictions.append(frame)
        metric_rows.append({"split": label, "cells": len(y), "MAPE_pct": 100*mean_absolute_percentage_error(y, pred),
                            "MAE_cycles": mean_absolute_error(y, pred),
                            "RMSE_cycles": np.sqrt(mean_squared_error(y, pred)), "R2": r2_score(y, pred)})
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_csv(out / "predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "supplementary_metrics.csv", index=False)
    cv = comparison.loc[comparison.candidate_id == spec["candidate_id"]].iloc[0]
    v, t = metrics.set_index("split").loc[["Valid", "Test"], "MAPE_pct"]
    performance = pd.DataFrame([
        ("Train (Batch 1 CV)", cv.CV_MAPE_pct, "%", f"5-fold mean; SD={cv.CV_std_pct:.6f}; 36 cells/18 groups"),
        ("Valid (Batch 1 Hold-out)", v, "%", f"{len(valid)} cells/{valid.policy.nunique()} groups"),
        ("Test (Batch 2)", t, "%", f"{len(test)} cells; final model frozen by B1 CV"),
        ("Gap (Train-Valid)", v-cv.CV_MAPE_pct, "%p", "Valid minus Train CV"),
        ("Gap (Valid-Test)", t-v, "%p", "Test minus Valid"),
        ("Gap (Target-Test)", t-9.1, "%p", "Test minus assignment target 9.1")], columns=["구분", "값", "단위", "비고"])
    performance.to_csv(out / "model_performance.csv", index=False)
    for key in ["life_group", "policy", "unseen_policy", "new_structure"]:
        pred_df.groupby(["split", key], observed=True).agg(cells=("cell_id", "size"),
            MAPE_pct=("APE_pct", "mean"), MAE_cycles=("absolute_error", "mean"),
            mean_residual=("residual", "mean")).reset_index().to_csv(out / f"errors_by_{key}.csv", index=False)
    pred_df.query("split == 'Test'").nlargest(8, "APE_pct").to_csv(out / "largest_test_errors.csv", index=False)
    importance = permutation_importance(model, valid[names], valid.cycle_life, n_repeats=20,
                                       random_state=SEED, scoring="neg_mean_absolute_percentage_error", n_jobs=1)
    pd.DataFrame({"feature": names, "MAPE_increase_pct": 100*importance.importances_mean,
                  "std_pct": 100*importance.importances_std}).sort_values("MAPE_increase_pct", ascending=False).to_csv(out / "valid_permutation_importance.csv", index=False)
    pipe = model.regressor_ if spec["log_target"] else model
    pre = pipe.named_steps["preprocess"]
    pruner = pre.named_transformers_["num"].named_steps["prune"]
    numeric = [n for n in names if n != "policy"]
    pd.DataFrame([{ "feature": n, "kept": i in pruner.keep_indices_,
                   "reason": "kept" if i in pruner.keep_indices_ else next(r[1] for r in pruner.removed_ if r[0] == i)}
                  for i,n in enumerate(numeric)]).to_csv(out / "final_feature_selection.csv", index=False)
    transformed = pre.transform(train[names])
    a = np.asarray(transformed.select_dtypes(include="number"), dtype=float)
    nonconstant = np.std(a, axis=0) > 1e-12
    a = a[:, nonconstant]
    z = (a-a.mean(axis=0))/a.std(axis=0)
    diag = {"numeric_features": a.shape[1], "numeric_condition_number": float(np.linalg.cond(z))}
    (out / "collinearity_diagnostics.json").write_text(json.dumps(diag, indent=2))
    reg = pipe.named_steps["model"]
    if hasattr(reg, "coef_"):
        pd.DataFrame({"feature": pre.get_feature_names_out(), "coefficient": reg.coef_}).to_csv(out / "model_coefficients.csv", index=False)
    joblib.dump(model, out / "final_model.joblib")
    return model, performance, metrics, pred_df
