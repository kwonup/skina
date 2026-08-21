# 외부 데이터 모델 성능 비교

- 평가 이미지: 138장
- 전처리: Resize(224×224) + ImageNet Normalize
- 평가지표의 Macro F1/Recall은 외부셋 정답 10개 클래스 기준

| Model | External Accuracy | Macro F1 | Macro Recall | 정답 수 |
|---|---:|---:|---:|---:|
| 이전 모델 | 21.74% | 0.1377 | 0.1694 | 30/138 |
| 최근 모델 | 60.87% | 0.5952 | 0.7263 | 84/138 |
| 개선폭 | +39.13%p | +0.4575 | +0.5569 | +54장 |

## 클래스별 Recall

| Class | Support | 이전 | 최근 | 변화 |
|---|---:|---:|---:|---:|
| actinic_keratosis | 1 | 0.0000 | 1.0000 | +1.0000 |
| basal_cell_carcinoma | 1 | 0.0000 | 1.0000 | +1.0000 |
| dermatofibroma | 1 | 0.0000 | 1.0000 | +1.0000 |
| hemangioma | 3 | 0.3333 | 0.6667 | +0.3333 |
| lentigo | 22 | 0.0455 | 0.2273 | +0.1818 |
| malignant_melanoma | 21 | 0.9048 | 0.6190 | -0.2857 |
| melanocytic_nevus | 21 | 0.3333 | 0.8571 | +0.5238 |
| seborrheic_keratosis | 22 | 0.0000 | 0.7273 | +0.7273 |
| squamous_cell_carcinoma | 20 | 0.0000 | 0.5500 | +0.5500 |
| wart | 26 | 0.0769 | 0.6154 | +0.5385 |

## 정오답 전환

- 이전 오답 → 최근 정답: 61장
- 이전 정답 → 최근 오답: 7장
- 두 모델 모두 정답: 23장
- 두 모델 모두 오답: 47장

이전 모델에만 존재하는 추가 출력 클래스로 예측한 경우도 오답으로 포함했습니다.

## 해석 시 주의사항

- 이전 checkpoint는 15개, 최근 checkpoint는 10개 클래스 모델입니다. 따라서 이 결과는 외부 10클래스 과제에서의 실제 Top-1 성능 비교이며, 클래스 구성 외의 학습 변경 효과만 분리한 통제 실험은 아닙니다.
- actinic_keratosis, basal_cell_carcinoma, dermatofibroma는 각 1장, hemangioma는 3장뿐이므로 해당 클래스 Recall과 Macro 지표의 표본 불확실성이 큽니다.

## 그래프

- `performance_comparison.png`: 핵심 지표와 클래스별 Recall 비교
- `confusion_matrix_previous.png`, `confusion_matrix_recent.png`: 모델별 혼동행렬
