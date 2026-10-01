"""Extract cell-level features using observations at cycles <= 100 only.

Full summaries are inspected solely for target/data auditing and descriptive EDA.
No target-derived or post-observation metadata is included in model features.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

FILES = {
    "B1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "B2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
}
GRID = np.linspace(2.1, 3.4, 500)
NUMERIC_FEATURES = [
    "delta_logvar", "QD_slope_10_100", "mean_chargetime", "mean_Tavg",
    "C1", "switch_SOC", "C2", "std_QD", "mean_IR", "IR_change",
    "delta_min", "delta_mean", "delta_range", "mean_Tmax",
]


def project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "data").is_dir() and (path / "day2/day2.md").is_file():
            return path
    raise FileNotFoundError("프로젝트 루트 또는 day2 폴더에서 실행하세요.")


def read_array(f: h5py.File, obj: h5py.Dataset) -> np.ndarray:
    value = obj[()]
    if h5py.check_dtype(ref=obj.dtype) is not None:
        value = np.concatenate([np.asarray(f[ref][()]).ravel() for ref in value.ravel()])
    return np.asarray(value).ravel()


def text_value(f: h5py.File, ref) -> str:
    return "".join(chr(int(v)) for v in read_array(f, f[ref]) if v)


def slope(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[valid], y[valid], 1)[0]) if valid.sum() >= 3 else np.nan


def load_features(root: Path):
    """Read only explicitly named Batch 1 and Batch 2 files, sequentially."""
    records, audits, cells, structures = [], [], {}, []
    for batch, filename in FILES.items():
        path = root / "data" / filename
        with h5py.File(path, "r") as f:
            b = f["batch"]
            structures.append({"batch": batch, "file": filename,
                               "size_bytes": path.stat().st_size,
                               "fields": list(b.keys()), "raw_cells": len(b["summary"])})
            for index in range(len(b["summary"])):
                cid = f"{batch}c{index}"
                sg, cg = f[b["summary"][index, 0]], f[b["cycles"][index, 0]]
                summary = pd.DataFrame({{"QDischarge": "QD", "QCharge": "QC"}.get(k, k):
                                        read_array(f, sg[k]) for k in sg})
                cycle = summary.cycle.to_numpy()
                assert np.array_equal(cycle, np.arange(1, len(summary) + 1)), cid
                assert len(summary) == len(cg["Qdlin"]), cid
                policy = text_value(f, b["policy_readable"][index, 0])
                life = float(read_array(f, f[b["cycle_life"][index, 0]])[0])
                voltage = read_array(f, f[b["Vdlin"][index, 0]])
                assert len(voltage) == 1000 and np.all(np.diff(voltage) < 0), cid
                assert voltage.min() <= GRID.min() and voltage.max() >= GRID.max(), cid
                early = summary.loc[summary.cycle.between(2, 100)].copy()
                qref = early.loc[early.cycle <= 10, "QD"]
                qref = qref[np.isfinite(qref) & (qref > 0)].median()
                bad = ~np.isfinite(early.QD) | (early.QD <= 0) | (early.QD > 1.3 * qref)
                early.loc[bad, "QD"] = np.nan
                match = re.match(r"^(\d+(?:\.\d+)?)C\((\d+(?:\.\d+)?)%\)-(\d+(?:\.\d+)?)C(.*)$", policy)
                rates = [float(match.group(j)) for j in (1, 2, 3)] if match else [np.nan] * 3
                curves = {}
                for n in (10, 100):
                    pos = np.flatnonzero(cycle == n)
                    if len(pos) == 1:
                        q = read_array(f, f[cg["Qdlin"][int(pos[0]), 0]])
                        if len(q) == len(voltage) and np.isfinite(q).all():
                            curves[n] = np.interp(GRID, voltage[::-1], q[::-1])
                delta = curves[100] - curves[10] if len(curves) == 2 else np.full(len(GRID), np.nan)
                variance = float(np.var(delta, ddof=0))
                signature = hashlib.sha256()
                # Exact first-100 fingerprints detect copied records; they do not prove physical independence.
                for key in ("QD", "IR", "Tavg"):
                    signature.update(np.asarray(summary.loc[summary.cycle <= 100, key], dtype="<f8").tobytes())
                signature.update(np.asarray(curves.get(10, []), dtype="<f8").tobytes())
                below = summary.loc[(summary.cycle >= 2) & (summary.QD > 0) & (summary.QD < 0.88), "cycle"]
                first_cross = float(below.iloc[0]) if len(below) else np.nan
                usable = np.isfinite(life) and life > 0 and set(range(1, 101)).issubset(cycle)
                reason = "included" if usable else ("missing_or_invalid_target" if not np.isfinite(life) or life <= 0 else "missing_first100")
                audit = {"cell_id": cid, "batch": batch, "index": index, "policy": policy,
                         "cycle_life": life, "n_cycles": len(summary), "qref": qref,
                         "last_QD": float(summary.QD.iloc[-1]), "first_below_0_88_after_cycle1": first_cross,
                         "label_equals_n_plus_1": bool(np.isfinite(life) and life == len(summary) + 1),
                         "label_matches_0_88_crossing": bool(np.isfinite(first_cross) and life == first_cross),
                         "observed_0_88_crossing": bool(len(below)),
                         "early_bad_QD_rows": int(bad.sum()), "early_summary_nan": int(early.isna().sum().sum()),
                         "has_first100": bool(set(range(1, 101)).issubset(cycle)),
                         "has_delta": len(curves) == 2, "v_min": float(voltage.min()),
                         "v_max": float(voltage.max()), "v_points": len(voltage),
                         "policy_parse_ok": match is not None,
                         "barcode_raw": ",".join(map(str, read_array(f, f[b["barcode"][index, 0]]))),
                         "channel_raw": ",".join(map(str, read_array(f, f[b["channel_id"][index, 0]]))),
                         "early_fingerprint": signature.hexdigest(), "included": usable, "reason": reason}
                audits.append(audit)
                cells[cid] = {"summary": summary, "delta": delta, "curves": curves, "qref": qref}
                if not usable:
                    continue
                ir = early.IR.where(early.IR > 0)
                rec = {"cell_id": cid, "batch": batch, "cycle_life": life, "policy": policy,
                       "C1": rates[0], "switch_SOC": rates[1], "C2": rates[2],
                       "delta_logvar": float(np.log10(variance)) if np.isfinite(variance) and variance > 0 else np.nan,
                       "delta_min": float(np.min(delta)), "delta_mean": float(np.mean(delta)),
                       "delta_range": float(np.ptp(delta)), "mean_QD": early.QD.mean(),
                       "initial_QD": qref, "std_QD": early.QD.std(ddof=1),
                       "QD_slope_10_100": slope(early.loc[early.cycle >= 10, "cycle"], early.loc[early.cycle >= 10, "QD"]),
                       "mean_IR": ir.mean(),
                       "IR_change": ir.loc[early.cycle.between(91, 100)].median() - ir.loc[early.cycle.between(2, 10)].median(),
                       "mean_Tavg": early.Tavg.mean(), "mean_Tmax": early.Tmax.mean(),
                       "mean_chargetime": early.loc[early.chargetime > 0, "chargetime"].mean()}
                records.append(rec)
        print(f"{batch}: selected fields loaded", flush=True)
    return (pd.DataFrame(records).replace([np.inf, -np.inf], np.nan),
            pd.DataFrame(audits), cells, structures)


def save_data_audit(root: Path, features, audits, structures):
    out = root / "day2/output"
    features.to_csv(out / "early_features.csv", index=False)
    audits.to_csv(out / "cell_audit.csv", index=False)
    quality = audits.groupby("batch").agg(
        raw_cells=("cell_id", "size"), used_cells=("included", "sum"),
        first100=("has_first100", "sum"), delta_available=("has_delta", "sum"),
        labels_n_plus_1=("label_equals_n_plus_1", "sum"),
        labels_matching_crossing=("label_matches_0_88_crossing", "sum"),
        observed_crossing=("observed_0_88_crossing", "sum"),
        early_bad_QD_rows=("early_bad_QD_rows", "sum")).reset_index()
    quality["excluded_cells"] = quality.raw_cells - quality.used_cells
    quality.to_csv(out / "batch_quality.csv", index=False)
    old = pd.read_csv(root / "day1/output/early_features.csv").query("batch in ['B1','B2']")
    merged = features.merge(old, on="cell_id", suffixes=("_new", "_old"), validate="one_to_one")
    comparisons = []
    for col in ["cycle_life", *NUMERIC_FEATURES]:
        a, b = merged[col + "_new"].to_numpy(), merged[col + "_old"].to_numpy()
        comparisons.append({"feature": col, "cells": len(merged),
                            "match": bool(np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)),
                            "max_abs_difference": float(np.nanmax(np.abs(a-b)))})
    pd.DataFrame(comparisons).to_csv(out / "day1_feature_recheck.csv", index=False)
    assert all(r["match"] for r in comparisons), "DAY 1/2 feature mismatch: inspect before fitting."
    copied = audits.groupby("early_fingerprint").filter(lambda g: g.batch.nunique() > 1)
    notes = {
        "loaded_batches": list(FILES), "structures": structures,
        "exact_cross_batch_copied_records": len(copied),
        "physical_cell_independence": "unresolved: opaque barcode/channel values are reused; no experimental log is available",
        "target_policy": "retain provided cycle_life; do not impute targets or replace B1 labels with assumed EOL",
        "label_limitation": "B1 labels are record length + 1 with no post-cycle1 measured QD<0.88; B2 finite labels match first measured QD<0.88. Different observation/label mechanisms limit EOL interpretation.",
        "canonical_paper_batch2": "2017-06-30; assignment/local Batch 2 is 2018-02-20; canonical continuation indices are not applied",
        "source": "https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m",
        "feature_data_window": "cycle 2..100; slope 10..100; delta 100-10; policies known before cycling",
        "QD_rule": "finite, >0, <=1.3*median(positive QD at cycles 2..10); no EOL lower-bound deletion",
        "voltage_rule": "provided descending Vdlin (3.5..2.0V); interpolate onto ascending 2.1..3.4V, 500 points without extrapolation",
        "units": {"QD": "Ah", "IR": "Ohm per dataset", "Tavg": "degrees Celsius per dataset",
                  "chargetime": "dataset summary units; not independently confirmed, reported as provided"},
    }
    (out / "data_audit.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    return quality, notes
