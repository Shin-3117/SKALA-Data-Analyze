"""Reproducible CLI; notebook calls the same functions without subprocesses."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo
import re

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from day2.src.features import load_features, save_data_audit, FILES
from day2.src.modeling import FEATURE_SETS, split_manifest, compare_candidates, evaluate_final
from day2.src.reporting import make_figures, export_feature_design, write_readme

PACKAGES = ["numpy", "pandas", "scipy", "h5py", "matplotlib", "scikit-learn", "catboost",
            "joblib", "threadpoolctl", "ipython", "ipykernel", "nbformat", "nbclient"]


def save_run_config(root, selected, figures):
    versions = {p: importlib.metadata.version(p) for p in PACKAGES}
    config = {"executed_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
              "python_version": platform.python_version(), "packages": versions,
              "random_state": 42, "loaded_batches": list(FILES), "source_files": FILES,
              "selected": selected, "figures": figures,
              "selection_rule": "minimum B1 training grouped-CV mean MAPE; ties: CV SD, candidate_id",
              "final_fit_population": "36 B1 training cells; hold-out excluded",
              "target": "provided cycle_life (positive), no target imputation",
              "cv": "GroupKFold(5), groups=original policy, hold-out from DAY 1 seed 42 plan",
              "fixed_model_parameters": {"RandomForest_ExtraTrees": {"n_estimators": 200, "random_state": 42, "n_jobs": 1},
                                         "CatBoost": {"iterations": 200, "learning_rate": 0.03, "random_seed": 42, "thread_count": 1},
                                         "ElasticNet": {"max_iter": 30000, "tol": 1e-6, "random_state": 42}},
              "cv_optimism": "reported winning CV was also used for candidate selection; not nested CV",
              "physical_identity_status": "unresolved: no trusted physical identifier or experiment log",
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((root/"day2/src").glob("*.py"))}}
    (root/"day2/output/run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (root/"requirements.txt").write_text("\n".join(f"{p}=={v}" for p,v in versions.items())+"\n")
    return config


def verify_run(root, features, train, valid, test, folds, selected, comparison, performance, predictions, model):
    checks = {}
    checks["only_B1_B2_used"] = set(features.batch) == {"B1", "B2"}
    checks["one_row_per_cell"] = features.cell_id.is_unique and predictions.cell_id.is_unique
    checks["no_holdout_policy_overlap"] = not bool(set(train.policy) & set(valid.policy))
    checks["group_folds_disjoint"] = all(not set(train.iloc[tr].policy) & set(train.iloc[va].policy) for tr,va in folds)
    covered = np.concatenate([va for _,va in folds])
    checks["OOF_covers_each_train_cell_once"] = np.array_equal(np.sort(covered),np.arange(len(train)))
    values = performance.set_index("구분")["값"]
    v = predictions.query("split == 'Valid'").APE_pct.mean()
    t = predictions.query("split == 'Test'").APE_pct.mean()
    cv_all = pd.read_csv(root/"day2/output/cv_fold_results.csv")
    cv_rows = cv_all.loc[cv_all.candidate_id == selected["candidate_id"]]
    cv = cv_rows.MAPE_pct.mean()
    checks["performance_and_predictions_match"] = bool(np.allclose([values.iloc[0],values.iloc[1],values.iloc[2]], [cv,v,t]))
    checks["gap_arithmetic_correct"] = bool(np.allclose(values.iloc[3:].to_numpy(),[v-cv,t-v,t-9.1]))
    checks["selected_minimum_CV"] = selected["candidate_id"] == comparison.iloc[0].candidate_id
    pipe = model.regressor_ if selected["log_target"] else model
    numeric = [n for n in FEATURE_SETS[selected["feature_set"]] if n != "policy"]
    imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    expected = train[numeric].median().fillna(0).to_numpy()
    checks["imputer_fit_on_training_only"] = bool(np.allclose(imputer.statistics_, expected, equal_nan=True))
    before = imputer.statistics_.copy()
    poison = test[FEATURE_SETS[selected["feature_set"]]].copy()
    for n in numeric:
        poison[n] = 999999.0
    pipe.named_steps["preprocess"].transform(poison)
    checks["transform_does_not_refit_on_test"] = bool(np.array_equal(before,imputer.statistics_))
    restored = joblib.load(root/"day2/output/final_model.joblib")
    names = FEATURE_SETS[selected["feature_set"]]
    checks["saved_model_prediction_roundtrip"] = bool(np.allclose(restored.predict(test[names]),model.predict(test[names])))
    checks["DAY1_features_recomputed_match"] = bool(pd.read_csv(root/"day2/output/day1_feature_recheck.csv")["match"].all())
    readme = (root/"README.md").read_text()
    sample_heads = [h.replace(" (sample)", "").strip() for h in re.findall(r"^## (.+)$",(root/"요구사항/sample.md").read_text(),re.M)]
    checks["README_sections_match_sample"] = re.findall(r"^## (.+)$",readme,re.M) == sample_heads
    missing=[]
    for link in re.findall(r"\]\(([^)]+)\)",readme):
        if link.startswith(("http://","https://")):
            continue
        path=root/unquote(link)
        if not path.exists() and path != root/"day2/output/verification.json":
            missing.append(link)
    checks["README_local_links_exist"] = not missing
    checks["path_independent_root_and_day2"] = (root/"day2/day2.md").is_file() and ROOT == root
    checks["all_predictions_finite"] = bool(np.isfinite(predictions.prediction).all())
    result={"checks":{k:bool(v) for k,v in checks.items()},"all_passed":bool(all(checks.values())),
            "missing_links":missing,"used_cells":{"B1":len(train)+len(valid),"B2":len(test)},
            "candidate_count":len(comparison),"fold_count":len(folds),
            "scientific_limitations":["different target/observation mechanisms between batches",
                                      "physical cell independence unresolved", "DAY1 explored B1 hold-out and B2",
                                      "winning CV used for selection; non-nested"],
            "external_actions":"no GitHub push/publication or Slack submission performed"}
    (root/"day2/output/verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    assert result["all_passed"], result
    return result


def run(root=ROOT):
    (root/"day2/output/image").mkdir(parents=True,exist_ok=True)
    features,audits,cells,structures=load_features(root)
    quality,notes=save_data_audit(root,features,audits,structures)
    train,valid,test,folds,manifest=split_manifest(root,features)
    selected,comparison=compare_candidates(root,train,folds)
    model,performance,metrics,predictions=evaluate_final(root,train,valid,test,folds,selected,comparison)
    figures=make_figures(root,features,cells,train,comparison,predictions)
    export_feature_design(root,selected)
    config=save_run_config(root,selected,figures)
    write_readme(root,features,quality,train,valid,test,selected,comparison,performance,metrics,predictions,config)
    verification=verify_run(root,features,train,valid,test,folds,selected,comparison,performance,predictions,model)
    print(performance.to_string(index=False))
    print("Verification:",verification["all_passed"])
    return verification


if __name__ == "__main__":
    run()
