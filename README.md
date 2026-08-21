<img src="service\web\public\skina_logo.png"/>

# skina
재구성된 피부 이미지 데이터로 10종 피부 병변을 분류하는 PyTorch 팀 프로젝트입니다. 네 모델을 같은 데이터 split과 baseline 조건으로 학습하고 Accuracy와 Macro F1을 비교합니다.

## Dataset

- AI-Hub Dataset 71864 피부종양 이미지 합성 데이터를 기반으로 재구성한 데이터
- 현재 processed 데이터 총 12,000장, 클래스당 1,200장
- 모델 입력 크기: 224×224
- 최종 split: train 10,000장 / validation 1,000장 / test 1,000장

이미지는 아래처럼 폴더명을 정답 라벨로 사용하도록 배치합니다. JSON 라벨은 보관만 하며 `ImageFolder` 학습에는 사용하지 않습니다.

```text
data/processed/
├── train/<class_name>/*.{jpg,jpeg,png}
├── val/<class_name>/*.{jpg,jpeg,png}
└── test/<class_name>/*.{jpg,jpeg,png}
```

## 10 Classes

```text
actinic_keratosis          basal_cell_carcinoma
dermatofibroma             hemangioma
lentigo                    malignant_melanoma
melanocytic_nevus          seborrheic_keratosis
squamous_cell_carcinoma    wart
```

`configs/class_names.json`이 프로젝트의 클래스 순서를 관리합니다. `ImageFolder`와 checkpoint의 `class_names`도 실행 시 이 순서와 일치하는지 검증합니다.

## Project Structure

```text
skina/
├── configs/
│   ├── cnn.json
│   ├── resnet18.json
│   ├── efficientnet_b0.json
│   └── mobilenet_v3.json
├── data/
│   ├── raw/{train,validation,labels}/
│   └── processed/{train,val,test}/
├── notebooks/01_eda.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── prepare_data.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── resnet18.py
│   │   ├── efficientnet_b0.py
│   │   └── mobilenet_v3.py
│   └── pipeline/
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       └── gradcam.py
├── outputs/
│   ├── models/
│   ├── plots/gradcam/
│   └── results/
├── sample/
├── docs/experiment_rules.md
├── requirements.txt
└── README.md
```

## Models

- `cnn`: 3개 convolution block과 Adaptive Average Pooling을 사용하는 Custom CNN
- `resnet18`: ImageNet pretrained ResNet18
- `efficientnet_b0`: ImageNet pretrained EfficientNet-B0
- `mobilenet_v3`: ImageNet pretrained MobileNetV3-Large

## Setup

Python 3.9 이상 환경을 권장합니다.

```bash
cd skina
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

W&B를 사용할 팀원은 최초 한 번 로그인합니다.

```bash
wandb login
```

## Configuration

모델별 JSON 파일에서 학습과 추론 설정을 관리합니다. 별도 YAML/Hydra 라이브러리는 사용하지 않습니다.

```json
{
  "model": "resnet18",
  "data": {
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 0,
    "seed": 42
  },
  "training": {
    "epochs": 15,
    "learning_rate": 0.0001,
    "optimizer": "Adam",
    "weight_decay": 0.0,
    "pretrained": true
  },
  "wandb": {
    "enabled": true,
    "entity": "sesac08",
    "project": "skina",
    "run_name": "resnet18_baseline"
  },
  "inference": {
    "top_k": 3,
    "checkpoint": "outputs/models/resnet18_best.pth"
  }
}
```

상대 checkpoint 경로는 프로젝트 루트를 기준으로 해석됩니다. 공통 baseline 비교가 끝나기 전에는 모델별 config 값을 서로 다르게 바꾸지 않습니다.

## Prepare Data

`prepare_data.py`는 최종 10개 클래스 원본만 받아 `raw/train`은 그대로 복사하고 `raw/validation`은 seed 42로 val/test에 절반씩 나눕니다. 현재 저장소의 `data/raw`는 이전 15클래스 데이터이므로 최종 10클래스 원본으로 교체하기 전에는 아래 명령을 실행하지 마세요. 특히 현재 processed 데이터를 보존하려면 `--overwrite`를 사용하지 않아야 합니다.

```bash
python -m src.data.prepare_data
```

기존 `data/processed`에 실제 데이터가 있으면 안전을 위해 중단합니다. 의도적으로 동일 규칙으로 다시 만들 때만 다음 옵션을 사용합니다.

```bash
python -m src.data.prepare_data --overwrite
```

DataLoader 확인은 선택 사항입니다. 학습할 때는 자동으로 생성됩니다.

```bash
python -m src.data.dataset
```

## Train

공통 baseline은 Epoch 15, Batch Size 32, Adam, Learning Rate 1e-4입니다. W&B project는 `skina`, run 이름은 `{model}_baseline`입니다.

```bash
python -m src.pipeline.train --config configs/cnn.json
python -m src.pipeline.train --config configs/resnet18.json
python -m src.pipeline.train --config configs/efficientnet_b0.json
python -m src.pipeline.train --config configs/mobilenet_v3.json
```

빠른 로컬 확인이나 설정 변경도 가능합니다.

```bash
python -m src.pipeline.train --config configs/resnet18.json --epochs 3 --batch-size 16 --no-wandb
```

CLI에서 지정한 값은 해당 실행에서만 config 값을 덮어쓰며 JSON 파일 자체는 변경하지 않습니다. 기존 `--model` 방식도 사용할 수 있지만 팀 실험에서는 config 방식을 권장합니다.

각 모델의 best checkpoint는 Validation Macro F1을 기준으로 `outputs/models/{model}_best.pth`에 저장됩니다.

## Evaluate

Test set은 모델 선택이나 튜닝에 사용하지 않고 best checkpoint가 정해진 후 아래 명령으로 한 번 평가합니다. Accuracy, Macro F1/Precision/Recall, 클래스별 classification report와 confusion matrix가 저장됩니다.

```bash
python -m src.pipeline.evaluate --config configs/resnet18.json
```

- 지표 JSON: `outputs/results/resnet18_metrics.json`
- confusion matrix: `outputs/plots/confusion_matrix_resnet18.png`

## Predict

```bash
python -m src.pipeline.predict --config configs/resnet18.json --image sample/example.jpg
```

config의 `top_k`에 따라 Softmax 확률이 높은 클래스를 출력합니다. 필요하면 `--top-k 5`처럼 이번 실행에서만 덮어쓸 수 있습니다.

## Grad-CAM

```bash
python -m src.pipeline.gradcam --config configs/resnet18.json --image sample/example.jpg
```

기본값은 예측 클래스를 설명하며 결과는 `outputs/plots/gradcam/`에 저장됩니다. 필요하면 클래스나 layer를 직접 정할 수 있습니다.

```bash
python -m src.pipeline.gradcam --config configs/resnet18.json --image sample/example.jpg \
  --target-class malignant_melanoma --target-layer layer4.1
```

## Team Roles

| 팀원 | 담당 모델       | 공통 담당            |
| ---- | --------------- | -------------------- |
| A    | Custom CNN      | Data / EDA           |
| B    | ResNet18        | Training / W&B       |
| C    | EfficientNet-B0 | Evaluation / Metrics |
| D    | MobileNetV3     | Inference / Grad-CAM |

각 팀원은 담당 모델 구현과 설정 파일을 관리합니다. `pipeline/`은 모든 모델이 동일한 학습·평가 방식을 사용하도록 공유합니다.

| 팀원 | 모델 구현                       | 설정 파일                      |
| ---- | ------------------------------- | ------------------------------ |
| A    | `src/models/cnn.py`             | `configs/cnn.json`             |
| B    | `src/models/resnet18.py`        | `configs/resnet18.json`        |
| C    | `src/models/efficientnet_b0.py` | `configs/efficientnet_b0.json` |
| D    | `src/models/mobilenet_v3.py`    | `configs/mobilenet_v3.json`    |

팀 공통 실험 규칙은 [`docs/experiment_rules.md`](docs/experiment_rules.md)를 따릅니다.
