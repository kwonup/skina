# 실험 규칙

## 공통 규칙

1. 모든 모델은 `python -m src.data.prepare_data`가 만든 동일한 Train / Val / Test split을 사용한다.
2. Seed는 42로 고정한다.
3. Input size는 224×224로 통일한다.
4. Baseline에서는 Epoch 15, Batch Size 32, Adam, Learning Rate 1e-4를 동일하게 사용한다.
5. Test set은 best model이 결정된 뒤 최종 평가에만 사용한다.
6. 모든 baseline 실험은 W&B project `skina`에 기록한다.
7. Best checkpoint는 Validation Macro F1 기준으로 저장한다.
8. 최종 모델 비교의 핵심 기준은 Accuracy와 Macro F1이다.
9. Baseline config는 네 모델에서 모델명·pretrained·출력 경로를 제외하고 동일하게 유지한다.

실험 조건을 변경한 추가 실험은 baseline과 구분되는 W&B run 이름을 사용하고, 변경한 값을 config에 남긴다.
