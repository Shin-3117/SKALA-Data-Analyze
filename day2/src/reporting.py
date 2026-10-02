"""Generate Batch 1/2 figures and sample.md-compatible result documentation."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/day2-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import GRID, NUMERIC_FEATURES
from .modeling import FEATURE_SETS

COLORS = {"B1": "#2563a7", "B2": "#d25c32"}


def md_table(frame, digits=3):
    frame = frame.copy()
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}" if np.isfinite(v) else "—"
        return str(v).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def make_figures(root, features, cells, train, comparison, predictions):
    image = root / "day2/output/image"
    image.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    created = []
    def save(fig, name):
        fig.savefig(image / name, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True, sharey=True, layout="constrained")
    for ax, batch in zip(axes, COLORS):
        f = features.loc[features.batch == batch]
        ax.hist(f.cycle_life, bins=np.arange(150, 2301, 100), color=COLORS[batch], edgecolor="white")
        ax.axvline(f.cycle_life.median(), color="black", ls="--", label=f"median {f.cycle_life.median():.1f}")
        ax.set(title=f"{batch}, n={len(f)}", xlabel="Provided cycle_life (cycles)", ylabel="Cells", xlim=(150,2300))
        ax.legend()
    save(fig, "eda_cycle_life.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True, layout="constrained")
    for ax, batch in zip(axes, COLORS):
        f = features.loc[features.batch == batch]
        for row in [f.loc[f.cycle_life.idxmin()], f.loc[f.cycle_life.idxmax()]]:
            s = cells[row.cell_id]["summary"]
            qref = cells[row.cell_id]["qref"]
            q = s.QD.where(np.isfinite(s.QD) & (s.QD > 0) & (s.QD <= 1.3*qref))
            ax.plot(s.cycle, q, lw=1.5, label=f"{row.cell_id}, label={row.cycle_life:.0f}")
        ax.axhline(0.88, color="black", ls="--", label="0.88 Ah reference")
        ax.axvspan(0, 100, color="#dddddd", alpha=.5, label="Input window <=100")
        ax.set(title=batch, xlabel="Recorded cycle", ylabel="QD (Ah)")
        ax.legend(fontsize=9)
    save(fig, "eda_degradation.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True, layout="constrained")
    for ax, batch in zip(axes, COLORS):
        f = features.loc[features.batch == batch]
        for name, mask, color in [("short <500", f.cycle_life < 500, "#d25c32"),
                                  ("middle 500-1000", f.cycle_life.between(500, 1000), "#7a7e83"),
                                  ("long >1000", f.cycle_life > 1000, "#2563a7")]:
            ids = f.loc[mask, "cell_id"]
            if not len(ids):
                continue
            curves = np.stack([cells[c]["delta"] for c in ids])
            median = np.nanmedian(curves, axis=0)
            lo, hi = np.nanquantile(curves, [.25, .75], axis=0)
            ax.plot(GRID, median, color=color, label=f"{name}, n={len(ids)}")
            ax.fill_between(GRID, lo, hi, color=color, alpha=.18)
        ax.set(title=batch, xlabel="Voltage (V)", ylabel="Q100 - Q10 (Ah)")
        ax.legend(fontsize=9)
    save(fig, "eda_delta_q.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True, layout="constrained")
    for ax, batch in zip(axes, COLORS):
        f = features.loc[features.batch == batch]
        ax.scatter(f.delta_logvar, f.cycle_life, c=COLORS[batch], alpha=.8)
        ax.set(title=f"{batch}, n={len(f)}, r={f.delta_logvar.corr(f.cycle_life):.3f}",
               xlabel="log10 Var(Delta Q / Ah)", ylabel="Provided cycle_life (cycles)")
    save(fig, "eda_delta_life.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), layout="constrained")
    for ax, batch in zip(axes, COLORS):
        summary = features.loc[features.batch == batch].groupby("policy").cycle_life.agg(["mean", "std", "size"]).sort_values("mean")
        ax.barh(np.arange(len(summary)), summary["mean"], xerr=summary["std"].fillna(0), color=COLORS[batch], alpha=.8)
        ax.set(yticks=np.arange(len(summary)), yticklabels=[f"{p} (n={int(r['size'])})" for p,r in summary.iterrows()],
               xlabel="Mean provided cycle_life (cycles)", title=f"{batch}: policy means; bars +/- SD")
        ax.tick_params(axis="y", labelsize=8)
    save(fig, "eda_policy.png")
    corr = train[NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(xticks=range(len(corr)), yticks=range(len(corr)), xticklabels=corr.columns,
           yticklabels=corr.index, title=f"B1 training portion: n={len(train)} (before pruning)")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    save(fig, "feature_correlation.png")
    families = comparison.groupby("family", sort=False).head(1).sort_values("CV_MAPE_pct")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.barh(families.family, families.CV_MAPE_pct, xerr=families.CV_std_pct, color="#2563a7", alpha=.85)
    ax.invert_yaxis()
    ax.set(xlabel="B1 grouped CV MAPE (%)", title="Best configuration per family; error bars = fold SD")
    save(fig, "model_comparison.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, layout="constrained")
    bound = max(predictions.cycle_life.max(), predictions.prediction.max())*1.06
    for ax, label in zip(axes, ["Train OOF", "Valid", "Test"]):
        d = predictions.loc[predictions.split == label]
        ax.scatter(d.cycle_life,d.prediction,c="#2563a7" if label != "Test" else "#d25c32",alpha=.85)
        ax.plot([0,bound],[0,bound],"k--",lw=1)
        ax.set(title=f"{label}: n={len(d)}, MAPE={d.APE_pct.mean():.2f}%", xlabel="Provided cycle_life (cycles)",
               ylabel="Predicted life (cycles)",xlim=(0,bound),ylim=(0,bound))
    save(fig, "actual_vs_predicted.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True, layout="constrained")
    for ax,label in zip(axes,["Train OOF","Valid","Test"]):
        d=predictions.loc[predictions.split==label]
        ax.scatter(d.cycle_life,d.residual,c="#2563a7" if label!="Test" else "#d25c32",alpha=.85)
        ax.axhline(0,c="black",ls="--")
        ax.set(title=f"{label}, n={len(d)}",xlabel="Provided cycle_life (cycles)",ylabel="Prediction - label (cycles)")
    save(fig, "residuals.png")
    group = predictions.query("split == 'Test'").groupby("life_group").agg(n=("cell_id","size"),mape=("APE_pct","mean"))
    fig,ax=plt.subplots(figsize=(7,3.8),layout="constrained")
    ax.bar(group.index,group.mape,color="#d25c32")
    for i,r in enumerate(group.itertuples()):ax.text(i,r.mape,f"n={r.n}",ha="center",va="bottom")
    ax.set(ylabel="Batch 2 MAPE (%)",title="Test error by provided-label range")
    save(fig,"test_error_groups.png")
    importance=pd.read_csv(root/"day2/output/valid_permutation_importance.csv").sort_values("MAPE_increase_pct")
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    ax.barh(importance.feature,importance.MAPE_increase_pct,xerr=importance.std_pct,color="#16816b")
    ax.axvline(0,c="black",lw=.8)
    ax.set(xlabel="Increase in hold-out MAPE (%p)",title="Permutation diagnostics: n=10; 20 repeats; no retuning")
    save(fig,"valid_importance.png")
    return created


def export_feature_design(root, spec):
    definitions = [
        ("delta_logvar","log10(var(Q100-Q10, ddof=0)), 2.1..3.4V, 500 points","10,100","log10(Ah^2)","B1 r=-0.887; primary signal","nonpositive variance/nonfinite -> NaN"),
        ("QD_slope_10_100","polyfit(cycle,QD,1)[0]","10..100","Ah/cycle","B1 r=0.524; initial slope","QD<=0,nonfinite,>1.3*qref -> NaN"),
        ("mean_chargetime","mean(positive chargetime)","2..100","dataset units","B1 r=0.577; charging process","nonpositive -> NaN"),
        ("mean_Tavg","mean(Tavg)","2..100","deg C","B1 r=-0.482; representative temperature","median imputation in fold"),
        ("C1","first C-rate parsed from policy","before cycling","C","B1 r=-0.580; policy numeric representation","parse errors -> NaN"),
        ("switch_SOC","transition SOC parsed from policy","before cycling","%","policy numeric representation","parse errors -> NaN"),
        ("C2","second C-rate parsed from policy","before cycling","C","policy numeric representation","parse errors -> NaN"),
        ("std_QD","sample std(QD), ddof=1","2..100","Ah","variability; optional group ablation","same QD quality rule"),
        ("mean_IR","mean(IR where IR>0)","2..100","Ohm per dataset","optional; zero is treated as missing","IR<=0 -> NaN"),
        ("IR_change","median(IR91:100)-median(IR2:10), IR>0","2..100","Ohm per dataset","optional change signal","IR<=0 -> NaN"),
        ("delta_min","min(Q100-Q10)","10,100","Ah","redundancy/offset ablation","nonfinite -> NaN"),
        ("delta_mean","mean(Q100-Q10)","10,100","Ah","redundancy/offset ablation","nonfinite -> NaN"),
        ("delta_range","ptp(Q100-Q10)","10,100","Ah","redundancy ablation","nonfinite -> NaN"),
        ("mean_Tmax","mean(Tmax)","2..100","deg C","temperature redundancy ablation","median imputation in fold"),
        ("policy","original policy string with suffix preserved","before cycling","category","one-hot/CatBoost vs numeric decomposition","unknown: one-hot ignore or CatBoost hash")]
    selection=pd.read_csv(root/"day2/output/final_feature_selection.csv").set_index("feature")
    selected=FEATURE_SETS[spec["feature_set"]]
    rows=[]
    for name,formula,window,unit,evidence,missing in definitions:
        status="not in winning feature set"
        if name in selected:
            status="selected" if name=="policy" or selection.loc[name,"kept"] else "removed by training-only correlation/constant rule"
        rows.append({"feature":name,"formula":formula,"observed_cycles":window,"unit":unit,
                     "EDA_basis":evidence,"final_status":status,"missing_rule":missing,
                     "selection_basis":"B1 training-policy CV; see feature_ablation.csv"})
    pd.DataFrame(rows).to_csv(root/"day2/output/feature_design.csv",index=False)


def write_readme(root, features, quality, train, valid, test, spec, comparison, performance, metrics, predictions, config):
    out = root / "day2/output"
    values = performance.set_index("구분")["값"]
    train_score = values["Train (Batch 1 CV)"]
    valid_score, test_score = values["Valid (Batch 1 Hold-out)"], values["Test (Batch 2)"]
    best = comparison.loc[comparison.candidate_id == spec["candidate_id"]].iloc[0]
    baseline = comparison.loc[comparison.family == "Median"].iloc[0].CV_MAPE_pct
    label_stats = features.groupby("batch").cycle_life.agg(["size", "median"])
    correlation = features.query("batch == 'B1'")["delta_logvar"].corr(features.query("batch == 'B1'")["cycle_life"])
    family = pd.read_csv(out/"model_family_best.csv")
    family_table = family[["family", "feature_set", "log_target", "params", "CV_MAPE_pct", "CV_std_pct"]].rename(
        columns={"family":"후보 모델", "feature_set":"피처군", "log_target":"로그 타깃", "params":"선택 파라미터",
                 "CV_MAPE_pct":"CV MAPE(%)", "CV_std_pct":"표준편차(%p)"})
    designs = pd.read_csv(out/"feature_design.csv")
    effective = designs.loc[designs.final_status == "selected"]
    selected_table = effective[["feature", "formula", "observed_cycles", "unit", "EDA_basis"]].rename(
        columns={"feature":"최종 피처", "formula":"계산식", "observed_cycles":"관측 사이클", "unit":"단위", "EDA_basis":"선정 근거"})
    full_features = FEATURE_SETS[spec["feature_set"]]
    perf_table = performance[["구분", "값"]].rename(columns={"값": "MAPE (%)"}).copy()
    perf_table["MAPE (%)"] = [
        f"{value:+.3f}" if label.startswith("Gap (") else f"{value:.3f}"
        for label, value in zip(perf_table["구분"], perf_table["MAPE (%)"])
    ]
    perf_table["비고"] = perf_table["구분"].map({
        "Train (Batch 1 CV)": f"5-fold 그룹 CV 평균; {len(train)}셀·{train.policy.nunique()}정책",
        "Valid (Batch 1 Hold-out)": f"{len(valid)}셀·{valid.policy.nunique()}정책",
        "Test (Batch 2)": f"{len(test)}셀; Batch 1 CV로 선택한 최종 모델 평가",
        "Gap (Train-Valid)": "(+) : 과적합 의심",
        "Gap (Valid-Test)": "(+) : 배치간 일반화 저하 의심",
        "Gap (Target-Test)": "Target : 원논문 9.1%",
    })
    auxiliary = metrics.rename(columns={"split":"구분", "cells":"셀 수", "MAPE_pct":"Pooled MAPE(%)",
                                        "MAE_cycles":"MAE(사이클)", "RMSE_cycles":"RMSE(사이클)"})
    errors = pd.read_csv(out/"errors_by_life_group.csv").query("split == 'Test'")
    error_table = errors[["life_group","cells","MAPE_pct","MAE_cycles","mean_residual"]].rename(
        columns={"life_group":"수명 구간", "cells":"셀 수", "MAPE_pct":"MAPE(%)", "MAE_cycles":"MAE(사이클)","mean_residual":"평균 예측−라벨(사이클)"})
    largest = predictions.query("split == 'Test'").nlargest(5,"APE_pct")
    top_table = largest[["cell_id","policy","cycle_life","prediction","APE_pct"]].rename(
        columns={"cell_id":"셀", "policy":"정책", "cycle_life":"실제 제공 라벨", "prediction":"예측", "APE_pct":"절대백분율오차(%)"})
    ablations = pd.read_csv(out/"feature_ablation.csv")
    ridge_abl = ablations.query("family == 'Ridge'")[["feature_set","corr_threshold","log_target","CV_MAPE_pct","CV_std_pct"]].sort_values("CV_MAPE_pct")
    ridge_abl = ridge_abl.rename(columns={"feature_set":"Ridge 피처군", "corr_threshold":"공선성 기준", "log_target":"로그 타깃", "CV_MAPE_pct":"CV MAPE(%)","CV_std_pct":"표준편차(%p)"})
    mean_bias = predictions.query("split == 'Test'").residual.mean()
    test_predictions = predictions.query("split == 'Test'")
    below_train = int((test.cycle_life < train.cycle_life.min()).sum())
    group_diagnostics = pd.read_csv(out/"errors_by_unseen_policy.csv").query("split == 'Test'").set_index("unseen_policy")
    structure_diagnostics = pd.read_csv(out/"errors_by_new_structure.csv").query("split == 'Test'").set_index("new_structure")
    condition = json.loads((out/"collinearity_diagnostics.json").read_text())["numeric_condition_number"]
    bias_text = "과대예측" if mean_bias > 0 else "과소예측"
    target_text = "목표보다 오차가 낮습니다" if test_score <= 9.1 else "목표에 미달합니다"
    outcome = f"Batch 1 그룹 CV {train_score:.3f}%, hold-out {valid_score:.3f}%, Batch 2 {test_score:.3f}%입니다. 과제 비교 목표 9.1%에 대해 {target_text}."
    contents = f'''# ESS 배터리 수명 예측

초기 100사이클의 측정 데이터로 제공된 배터리 총 수명 `cycle_life`를 회귀 예측하고, 별도 실험 배치에서 일반화 성능을 평가합니다. ESS 교체·점검 계획에 활용할 가능성과 데이터·모델의 한계를 함께 검토합니다.

**결과:** 최종 모델은 {spec['family']}이며 {outcome}

> **라벨 해석의 한계:** Batch 1의 46셀 모두 수명 라벨이 기록 길이+1이며, 저장된 기록에서 cycle 2 이후 QD<0.88Ah 도달은 없습니다. Batch 2의 유효 39셀은 제공 라벨과 최초 0.88Ah 미만 도달 사이클이 일치합니다. 아래 수치는 이 서로 다른 방식의 **제공 라벨에 대한 예측 오차**이며, 실제 EOL 수명을 동일 조건으로 검증한 결과로 단정할 수 없습니다.

## 프로젝트 개요

- 데이터셋: 과제 제공 MIT–Stanford Battery Dataset, Severson et al. (2019)
- 학습 데이터: Batch 1 (2017-05-12), 46셀 중 학습 36셀·검증 10셀
- 평가 데이터: Batch 2 (2018-02-20), 원본 47셀 중 라벨 결측 8셀을 제외한 39셀
- 태스크: **Regression — 초기 100사이클에서 제공된 총 `cycle_life` 예측**
- 타깃: 제공된 총 사이클 수입니다. 잔여 수명(RUL) 자체가 아니며 관측 기간 이후의 실제 EOL 라벨인지 추가 확인이 필요합니다.
- 주평가 지표: MAPE(%), 과제 비교 목표 9.1%
- 사용 범위: **Batch 1·2만 사용하며 Batch 3은 로딩·피처 생성·평가에서 제외했습니다.**
- 상세 작업 기준: [day2/day2.md](day2/day2.md)

## 파일 구조

```text
├── README.md
├── requirements.txt                 # DAY 2 실제 환경의 패키지·버전
├── data/
│   └── README.md                    # 원본 획득·배치 안내
├── 요구사항/                        # 과제·평가 기준·샘플
├── day1/                            # 기존 EDA·설계 산출물 보존
└── day2/
    ├── day2.md
    ├── day2.ipynb                   # 실행한 분석 코드·출력·해석
    ├── src/
    │   ├── features.py              # 원본 로딩·초기 피처·라벨 감사
    │   ├── modeling.py              # 그룹 CV·전처리·후보·평가
    │   ├── reporting.py             # 그래프·피처표·README 생성
    │   └── train.py                 # 전체 실행·정합성 검증
    └── output/
        ├── early_features.csv
        ├── cell_audit.csv
        ├── data_audit.json
        ├── feature_design.csv
        ├── split_manifest.csv
        ├── model_comparison.csv
        ├── cv_fold_results.csv
        ├── feature_ablation.csv
        ├── model_performance.csv
        ├── predictions.csv
        ├── final_model.joblib
        ├── run_config.json
        ├── verification.json
        └── image/
```

## 환경 설정

실행 환경은 Python {config['python_version']}입니다. 패키지 버전은 [requirements.txt](requirements.txt)에 기록했습니다. 원본 데이터를 배치한 프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python day2/src/train.py
```

기존 `.venv`가 있다면 해당 환경을 그대로 사용할 수 있습니다. `day2` 폴더에서는 `../.venv/bin/python src/train.py`로 같은 출력 위치에 실행합니다. 노트북은 `.venv`를 커널로 선택하고 위에서 아래로 실행합니다. CLI와 노트북은 같은 모듈을 호출하며 `day2/output/` 및 이 README의 결과를 갱신합니다.

[데이터 획득·배치 안내](data/README.md)에 따라 아래 두 원본 파일을 `data/`에 놓습니다. 원본 `.mat`는 GitHub 업로드 대상에서 제외합니다.

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`

## EDA

- **Cycle Life 분포:** 제공 라벨 중앙값은 Batch 1 {label_stats.loc['B1','median']:.1f}사이클(n=46), Batch 2 {label_stats.loc['B2','median']:.1f}사이클(n=39)입니다. Batch 1의 550 미만 표본은 1셀로 분류 검증이 불안정해 회귀를 선택했습니다. Batch 2의 짧은 라벨 분포와 서로 다른 라벨 생성 방식을 구분해 해석합니다.

![Batch 1·2 제공 수명 라벨 분포](day2/output/image/eda_cycle_life.png)

- **열화 곡선 분석:** 배치별 최소·최대 제공 라벨의 대표 셀을 비교했습니다. 회색 영역만 입력 관측 기간입니다. 0.88Ah 선은 원논문 로더의 절대 용량 참조 기준이며 각 셀 초기 용량의 80%와 동일하다고 표시하지 않습니다. 전체 기록의 knee·종료 용량·기록 길이는 모델 입력에서 제외했습니다.

![배치별 대표 셀의 방전 용량과 입력 관측 기간](day2/output/image/eda_degradation.png)

- **ΔQ(V) 곡선 분석:** `Q100−Q10`을 확인된 Vdlin 축(3.5→2.0V)에서 공통 2.1~3.4V·500점으로 보간했습니다. 띠는 그룹의 IQR입니다. Batch 1에는 <500 라벨 셀이 없어 해당 그룹 곡선이 없습니다. Batch 1 ΔQ 로그 분산–라벨 Pearson r={correlation:.3f}이며, 이를 후보 선정 근거로 삼았습니다.

![수명 그룹별 초기 ΔQ 곡선](day2/output/image/eda_delta_q.png)

![초기 ΔQ 로그 분산과 제공 라벨의 관계](day2/output/image/eda_delta_life.png)

- **충전 속도(C-rate)와 수명의 관계:** 정책별 평균 라벨·표본 수를 함께 확인했습니다. 오차막대는 표준편차이며 1셀 정책의 막대는 변동성 추정이 아닙니다. 충전 조건과 다른 실험 조건이 함께 달라지므로 정책 차이를 인과효과로 단정하지 않습니다. 수치 분해와 정책 범주형 처리 후보를 CV로 비교했습니다.

![배치별 충전 정책의 제공 수명 라벨 평균과 표본 수](day2/output/image/eda_policy.png)

- **추가 확인 — 다중공선성:** 아래 행렬은 Batch 1 학습 부분 36셀의 초기 피처 상관입니다. 각 CV fold 안에서 대표 피처 우선순위에 따라 공선성을 축소하고, Ridge 중복 피처군의 축소 유무도 비교했습니다. 결측 대치·선택·스케일링은 해당 학습 fold에만 fit했습니다.

![학습 부분 초기 피처의 상관관계](day2/output/image/feature_correlation.png)

## Modeling

### 피처 엔지니어링 전략

입력 후보군은 `{spec['feature_set']}`이며 전달된 변수는 `{', '.join(full_features)}`입니다. 아래 표는 학습 부분에 최종 fit한 전처리에서 유지된 피처입니다. CV fold마다 공선성에 따라 유지 피처가 달라질 수 있으며 최종 피처만 먼저 선택해 CV를 다시 계산하지 않았습니다.

{md_table(selected_table)}

QD는 비유한·0 이하·초기 2~10사이클 양수 중앙값의 1.3배 초과만 결측 처리하고, EOL 이하 용량을 일괄 제거하지 않았습니다. ΔQ 분산은 `ddof=0`이며 0·비유한값은 결측 처리합니다. 수치 결측은 학습 fold 중앙값으로 대치하고 상수·고상관 피처를 제거합니다. 기본 공선성 기준은 `|r|≥0.85`, 중복 피처 비교의 1.01은 상관 기반 제거를 비활성화한 설정입니다. 정책 문자열은 One-hot의 미관측 범주 무시 또는 CatBoost 범주 처리로 대응합니다.

{md_table(ridge_abl)}

위 비교는 각 피처군 내에서 CV로 고른 최선 설정이며 통제된 단일 파라미터 효과의 인과실험은 아닙니다. 중복 피처를 남긴 Ridge의 CV가 더 낮은 경우도 있어 제거만으로 성능이 항상 개선된다고 결론 내리지 않습니다. 최종 모델의 4개 수치 피처는 상관 기준으로 제거되지 않았고, 학습 부분에서 대치 후 표준화한 행렬의 조건수는 {condition:.3f}입니다. 전체 후보는 [model_comparison.csv](day2/output/model_comparison.csv), 피처 정의·채택/제외 근거는 [feature_design.csv](day2/output/feature_design.csv), 최종 축소 결과는 [final_feature_selection.csv](day2/output/final_feature_selection.csv)에 기록했습니다.

### 모델 선택 및 근거

- 후보 모델: 중앙값 기준, Ridge/Elastic Net, Random Forest/Extra Trees, CatBoost
- 최종 모델: **{spec['family']}**, 후보 `{spec['candidate_id']}`
- 최종 설정: `{json.dumps(spec['params'], sort_keys=True)}`, 로그 타깃 `{spec['log_target']}`, 공선성 기준 `{spec['corr_threshold']}`
- 고정 설정: Random Forest/Extra Trees는 200개 트리와 random_state=42를 사용했습니다. 얕은 깊이·최소 leaf 표본 수로 소표본 과적합을 제한하고 로그 타깃의 효과는 원래 단위 MAPE로 비교했습니다.
- 선택 이유: 사전 정의한 {len(comparison)}개 설정 중 **Batch 1 그룹 CV 평균 MAPE가 가장 낮았습니다**. 기준 모델 {baseline:.3f}% 대비 {baseline-train_score:.3f}%p 개선했고 fold 표준편차는 {best.CV_std_pct:.3f}%p입니다. Valid·Batch 2 결과로 모델을 바꾸지 않았습니다.

{md_table(family_table)}

선형 모델은 강한 ΔQ 신호·소표본·공선성에 대응하는 규제 후보, 트리는 비선형·상호작용 비교 후보, CatBoost는 정책 범주형 처리 후보로 선정했습니다. Ridge alpha는 0.01~100, Elastic Net alpha는 0.01/0.1/1과 l1_ratio 0.2/0.8, 트리 깊이는 2/4·최소 leaf 표본 수는 3/6, CatBoost 깊이는 2/4·l2_leaf_reg는 3/10을 비교했습니다. 로그 타깃은 역변환 후 MAPE를 계산했습니다. 정확한 전체 조합은 [search_space.json](day2/output/search_space.json)에 있습니다.

![후보별 최선 설정의 그룹 CV MAPE와 fold 표준편차](day2/output/image/model_comparison.png)

기존 seed=42 정책 hold-out 계획을 유지했습니다. 학습 {len(train)}셀·{train.policy.nunique()}정책에서 5-fold GroupKFold로 선택하고, 별도 Valid {len(valid)}셀·{valid.policy.nunique()}정책과 Batch 2 {len(test)}셀을 평가했습니다. 정책 그룹은 학습·검증 및 각 CV fold 사이에서 겹치지 않습니다. 동일 물리 셀의 독립성은 별도 실험 로그가 없어 완전히 보장하지 못합니다. [분할 목록](day2/output/split_manifest.csv)과 [fold별 점수](day2/output/cv_fold_results.csv)를 저장했습니다.

최종 모델은 hold-out을 제외한 Batch 1 학습 36셀에 fit했습니다. 전처리 → 모델의 동일 파이프라인으로 Valid·Test를 예측하고 Train은 OOF 예측을 저장했습니다. [최종 모델](day2/output/final_model.joblib)과 [실행 설정](day2/output/run_config.json)으로 재현할 수 있습니다. CV 점수는 후보 선택에도 사용됐으므로 검색으로 인한 낙관성이 남습니다.

## 성능 결과

`MAPE(%) = 100 × mean(|y−ŷ|/|y|)`이며 타깃은 제공 `cycle_life`입니다.

{md_table(perf_table)}

{outcome} 표의 Gap은 퍼센트포인트(%p)이며 각각 Valid−Train, Test−Valid, Test−Target으로 계산합니다. 양수는 오차 증가·목표 미달입니다. Train–Valid 차이는 과적합뿐 아니라 hold-out의 수명·정책 이동도 반영합니다. Valid 라벨 범위는 {valid.cycle_life.min():.0f}~{valid.cycle_life.max():.0f}, 학습은 {train.cycle_life.min():.0f}~{train.cycle_life.max():.0f}입니다.

{md_table(auxiliary)}

Train 주지표는 5개 fold MAPE의 단순 평균이고 위 보조 표는 셀별 OOF를 합친 MAPE입니다. fold 크기가 달라 두 값은 다를 수 있습니다. 원논문 로더의 Batch 2는 2017-06-30이고 과제·로컬 Batch 2는 2018-02-20입니다. 원논문 분할·라벨·전처리와 동일 조건의 재현이라고 주장하지 않으며 9.1%는 과제 비교 기준으로 사용했습니다.

![Train OOF·Valid·Test 실제 제공 라벨과 예측값](day2/output/image/actual_vs_predicted.png)

## 오류 분석

Batch 2 평균 예측−라벨은 {mean_bias:.3f}사이클로 {bias_text} 방향입니다. 수명 구간별 성능은 다음과 같습니다.

{md_table(error_table)}

![수명 라벨 구간별 Batch 2 오차](day2/output/image/test_error_groups.png)

절대백분율오차가 큰 Batch 2 셀의 상위 5개입니다.

{md_table(top_table)}

큰 오차 셀의 정책·측정 구조·라벨 구간과 평균 잔차를 [정책별 오류](day2/output/errors_by_policy.csv), [구조별 오류](day2/output/errors_by_new_structure.csv), [미관측 정책별 오류](day2/output/errors_by_unseen_policy.csv)에서 확인합니다. 라벨 정의·분포 이동·미관측 정책은 가능한 원인 가설이며 단독 원인으로 확정하지 않습니다. 개선 방향은 일관된 실제 EOL 라벨 확보, 더 다양한 학습 정책·짧은 수명 셀 수집, 별도 배치 재검증입니다. 이 분석 후 Batch 2에 맞춰 재튜닝하지 않았습니다.

**구체적인 오류 패턴:** 상위 5개 셀은 모두 500사이클 미만 라벨·미관측 정책·`newstructure` 접미사 없음이라는 공통점이 있습니다. 학습 최소 라벨은 {train.cycle_life.min():.0f}인데 Batch 2의 {below_train}/{len(test)}셀이 그보다 짧으며, Batch 2 예측 범위는 {test_predictions.prediction.min():.1f}~{test_predictions.prediction.max():.1f}사이클로 짧은 셀의 수명을 과대예측했습니다. Random Forest의 leaf 평균과 로그 역변환은 학습 타깃 범위 밖으로 수명을 외삽하지 못하는 구조입니다. 따라서 학습에 없는 짧은 수명 구간과 모델의 외삽 한계가 맞물렸음을 확인할 수 있습니다.

미관측 정책 셀의 MAPE는 {group_diagnostics.loc[True,'MAPE_pct']:.3f}%, 학습에 등장한 정책 셀은 {group_diagnostics.loc[False,'MAPE_pct']:.3f}%로 비슷합니다. 미관측 정책만으로 큰 오차를 설명하지 않습니다. `newstructure`가 없는 {int(structure_diagnostics.loc[False,'cells'])}셀은 MAPE {structure_diagnostics.loc[False,'MAPE_pct']:.3f}%, 접미사가 있는 {int(structure_diagnostics.loc[True,'cells'])}셀은 {structure_diagnostics.loc[True,'MAPE_pct']:.3f}%입니다. 구조 그룹은 수명 분포도 다르므로 이 차이를 구조 변경의 인과효과로 해석하지 않습니다.

![제공 라벨 대비 잔차](day2/output/image/residuals.png)

![고정 hold-out의 permutation 중요도](day2/output/image/valid_importance.png)

중요도는 최종 모델 확정 후 Valid 10셀에서 20회 순열로 산출한 설명용 진단입니다. 막대는 MAPE 증가, 오차막대는 반복 표준편차이며 음수도 가능합니다. 소표본·상관 피처 영향이 있어 인과적 기여도나 안정적인 순위로 단정하지 않습니다.

## ESS 도메인 해석

- 활용 가능성: 초기 열화 신호를 점검·추가 실험 대상 선정과 교체 계획의 참고 정보로 검토할 수 있습니다. 제공 라벨의 차이와 외부 성능을 고려할 때 자동 교체 시점 결정에 바로 적용할 근거는 부족합니다.
- 개발 한계: 소표본, 학습·검증·외부 배치의 분포 차이, Batch 1 종료+1 라벨, 물리 셀 중복 식별 불확실성이 있습니다. DAY 1에서 Batch 1 전체와 Batch 2를 탐색했다는 평가 독립성의 한계도 남습니다.
- 실 배포에 필요한 검증: 동일한 명목 용량·EOL 기준과 미완주/검열 여부, 실제 셀 식별자·실험 로그를 확보하고, 실제 ESS의 온도·부하·팩 조건에서 외부 검증과 예측 불확실성 평가를 해야 합니다. 실제 EOL 라벨 확보 또는 검열을 반영한 수명 모델은 후속 연구 과제입니다.
- 운영 범위: 초기 100사이클 관측이 필요하며 이번 외부 평가는 Batch 2에 한정합니다. 비용 절감·현장 운영 개선을 달성했다고 주장하지 않습니다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- [원논문 데이터 로딩 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m): 0.88Ah 도달 여부에 따른 라벨 생성, 원논문 배치 구성·분할·이어진 측정 처리 확인
- [과제 제공 데이터셋](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)
- [과제 요구사항](요구사항/요구사항%20문서.md), [추가 평가 기준](요구사항/추가적인_평가기준.md), [README 샘플](요구사항/sample.md)
- [DAY 1 분석](day1/day1.ipynb), [데이터 감사](day2/output/data_audit.json), [실행 검증](day2/output/verification.json)

## 팀 구성

- 신현중 (울산 3반): EDA, 피처 엔지니어링, 파이프라인·후보 모델 개발, 성능 평가(Batch 2), 오류·도메인 해석
'''
    (root / "README.md").write_text(contents, encoding="utf-8")
