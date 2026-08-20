<img src="skina_logo.png"/>

# skina

AI-Hub 합성 이미지와 실제 피부 병변 이미지로 10종 피부 병변을 분류하는 PyTorch 팀 프로젝트입니다. 네 모델을 같은 데이터 split과 baseline 조건으로 학습하고 Accuracy와 Macro F1을 비교합니다.

## Dataset

- Synthetic train: AI-Hub 클래스별 원본 800장 중 seed 42로 500장 선택
- Real: 클래스별 train 500장, validation 100장, test 100장
- 모델 입력 크기: 224×224
- 최종 split: train 10,000장 / validation 1,000장 / test 1,000장
- Train은 클래스마다 Synthetic 500장 + Real 500장으로 구성

Synthetic 원본과 Real 원본은 수정하지 않습니다. 준비 스크립트는 두 원본에서 파일을 복사하고 이름을 바꿔 새 `processed` 데이터셋을 만듭니다.

```text
data/raw/
├── train/<class_name>/*          # Synthetic 원본, 클래스당 800장
├── validation/<class_name>/*     # 기존 Synthetic validation/test용 원본(이번 split에는 미사용)
└── labels/                       # 보관만 함

<real_root>/
└── <한글 클래스 폴더>/*.png      # 새 Real 원본
```

## 10 Classes

```text
actinic_keratosis             basal_cell_carcinoma
dermatofibroma                hemangioma
lentigo                       malignant_melanoma
melanocytic_nevus             squamous_cell_carcinoma
seborrheic_keratosis          wart
```

클래스 index는 `data/processed/class_names.json`의 순서로 고정되며 checkpoint의 `class_names`에도 저장됩니다.

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
│   └── processed/
│       ├── {train,validation,test}/
│       ├── predict_pool/{real,synthetic}/
│       ├── class_names.json
│       ├── dataset_manifest.csv
│       └── split_summary.csv
├── notebooks/01_eda.ipynb
├── scripts/prepare_dataset.py
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

seed 42로 Synthetic train 500장과 Real train/validation/test를 분할합니다. Real 원본이 부족한 `actinic_keratosis`와 `malignant_melanoma`는 train 원본에만 보수적인 증강을 적용해 Real train을 500장으로 맞춥니다. 사용하지 않은 Synthetic train 및 Real 원본은 학습/공식 평가에서 제외되는 `predict_pool`로 복사합니다.

```bash
python scripts/prepare_dataset.py --real-root "C:\path\to\10개클래스_합산"
```

현재 PC에서는 `--real-root`를 생략하면 이 작업에 사용한 다운로드 폴더를 기본값으로 사용합니다. 다른 환경에서는 반드시 실제 경로를 지정하세요. 기존 `data/processed`가 있으면 안전을 위해 중단하며, 검증된 새 빌드로 의도적으로 교체할 때만 `--overwrite`를 사용합니다.

```bash
python scripts/prepare_dataset.py --real-root "C:\path\to\10개클래스_합산" --overwrite
```

스크립트는 `dataset_manifest.csv`, `split_summary.csv`, `class_names.json`을 만들고 SHA-256 기준 공식 split 중복, 증강 parent 누수, Predict Pool 격리를 검사합니다. 기존 호환 명령 `python -m src.data.prepare_data`도 같은 스크립트를 실행합니다.

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


### 공통 환경 설치

python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

### NVIDIA GPU 사용

python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu118

### CPU 또는 macOS 사용

python -m pip install torch torchvision
