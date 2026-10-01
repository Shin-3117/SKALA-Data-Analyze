# 원본 데이터 준비

[과제 제공 Kaggle 데이터셋](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 다음 두 파일을 내려받아 이 폴더에 배치합니다.

| 배치 | 파일 | DAY 2 용도 |
| --- | --- | --- |
| Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 학습·그룹 CV·고정 hold-out |
| Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 최종 외부 평가 |

원본 `.mat` 파일은 대용량이며 GitHub 업로드 대상에서 제외합니다. `features.py`는 위 두 이름만 명시적으로 읽으며 폴더의 다른 파일은 읽지 않습니다. Batch 3과 varcharge 파일은 DAY 2에서 사용하지 않습니다.

MATLAB v7.3 HDF5 구조를 `h5py`로 읽습니다. 모델 입력은 초기 100사이클 이내의 요약값·10/100번 `Qdlin`과 실험 시작 시의 정책입니다. 전체 summary는 라벨 감사와 설명용 그래프에만 사용합니다.

로컬 Batch 1 라벨은 기록 길이+1이고, 유효 Batch 2 라벨은 최초 QD<0.88Ah 도달과 일치합니다. 원논문 로더의 Batch 2는 2017-06-30으로 과제 지정 파일과 다릅니다. 따라서 원논문의 이어진 셀 인덱스 결합·제외 목록을 이 파일에 임의로 적용하지 않습니다. 상세 감사 결과는 [data_audit.json](../day2/output/data_audit.json)에 있습니다.
