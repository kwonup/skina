# skina 서비스 구현 지시서 for Codex

## 0. 문서 목적

현재 `skina` 프로젝트의 학습/평가/추론 파이프라인을 유지한 상태에서, 최종 선정된 PyTorch 모델을 실제 웹 서비스로 연결한다.

이번 서비스 구현의 목표는 **복잡한 기능을 추가하는 것**이 아니라 아래 흐름을 안정적으로 완성하는 것이다.

```text
사용자 이미지 업로드
        ↓
Next.js
        ↓
FastAPI POST /predict
        ↓
PyTorch 최종 모델
        ↓
전처리 / 추론 / Softmax
        ↓
Top-3 결과
        ↓
예측 병변의 사전 정의 정보
        ↓
선택적으로 Grad-CAM
        ↓
Next.js 결과 화면 표시
```

프로젝트의 핵심은 웹서비스 자체가 아니라 다음 End-to-End 흐름을 보여주는 것이다.

```text
Data
→ Training
→ W&B
→ Evaluation
→ Final Model Selection
→ Inference
→ XAI
→ FastAPI
→ Next.js
→ Docker
```

---

# 1. 가장 중요한 변경사항

기존 프로젝트는 피부 병변 **15개 클래스**를 대상으로 설계되었지만, 추가 데이터 학습 및 데이터셋 재구성 이후 **최종 서비스 대상 클래스는 10개**로 변경되었다.

따라서 서비스 구현 전 프로젝트 전체에서 기존의 `15 classes`, `num_classes=15`, 15개 클래스명 하드코딩 등이 남아 있는지 확인하고 수정한다.

## 반드시 확인할 것

- 모델의 `num_classes`
- classifier / fc output dimension
- `class_names.json`
- config 파일
- `predict.py`
- `evaluate.py`
- `gradcam.py`
- API 응답
- 프론트엔드 병변 목록
- README / 문서
- 테스트 코드
- 샘플 응답 JSON
- 병변 정보 JSON

**10개 클래스 이름은 새로 임의 작성하거나 추측하지 말고, 현재 프로젝트의 실제 데이터셋 폴더 또는 현재 사용 중인 `class_names.json` / config를 source of truth로 사용한다.**

가능하면 클래스 목록은 한 곳에서만 관리하고 학습/평가/추론/서비스가 동일한 클래스 순서를 사용하도록 한다.

예:

```text
configs/class_names.json
```

또는 현재 프로젝트에서 이미 사용 중인 클래스 관리 방식을 유지한다.

---

# 2. 구현 원칙

이번 작업에서는 기존 프로젝트를 크게 리팩터링하지 않는다.

특히 이미 구현되어 있는 아래 코드가 있다면 최대한 재사용한다.

```text
src/pipeline/train.py
src/pipeline/evaluate.py
src/pipeline/predict.py
src/pipeline/gradcam.py
```

서비스 전용으로 학습 코드를 복사해 새로 구현하지 않는다.

핵심 원칙:

1. 기존 inference 전처리 로직을 재사용한다.
2. 학습 시 사용한 Resize / Normalize 설정과 서비스 추론 설정을 반드시 동일하게 유지한다.
3. 클래스 순서도 학습 시와 동일하게 유지한다.
4. 최종 모델 checkpoint는 서버 시작 시 1회만 로드한다.
5. API 요청마다 모델을 다시 로드하지 않는다.
6. 로그인, 회원가입, DB, 관리자 기능은 추가하지 않는다.
7. 서비스는 stateless 구조로 유지한다.
8. 기능보다 안정적인 End-to-End 동작을 우선한다.
9. 불필요한 Clean Architecture, Repository Pattern, CQRS 등은 도입하지 않는다.
10. 기존 코드 스타일과 폴더 구조를 최대한 존중한다.

---

# 3. 권장 최종 프로젝트 구조

현재 프로젝트 구조를 먼저 확인한 뒤, 아래 구조를 참고하여 최소한으로 확장한다.

```text
skina/
│
├── configs/
│   └── ...
│
├── data/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── docs/
│   └── ...
│
├── outputs/
│   ├── models/
│   │   └── best_model.pth
│   ├── plots/
│   └── results/
│
├── src/
│   ├── data/
│   ├── models/
│   └── pipeline/
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       └── gradcam.py
│
├── service/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── inference.py
│   │   ├── schemas.py
│   │   └── lesion_info.json
│   │
│   └── web/
│       ├── app/
│       ├── components/
│       ├── lib/
│       ├── public/
│       ├── package.json
│       └── ...
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

현재 repository 구조와 충돌한다면 기존 구조를 우선한다.

---

# 4. Phase 1 — 현재 프로젝트 분석

코드를 수정하기 전에 먼저 repository를 분석한다.

## 확인할 내용

### 모델 관련

- 현재 모델 구현 위치
- 최종 모델 architecture
- checkpoint 저장 형식
- checkpoint가 `state_dict`인지 전체 model object인지
- 최종 classifier output dimension
- 현재 실제 클래스 수
- 클래스 index → class name 매핑 방법

### 추론 관련

`predict.py`에서 다음을 파악한다.

- 이미지 로딩 방식
- Resize 크기
- Normalize mean/std
- RGB 변환 여부
- device 선택 방식
- softmax 적용 방식
- Top-K 계산 방식
- 모델 로딩 방식

### Grad-CAM 관련

`gradcam.py`에서 다음을 파악한다.

- 어떤 library를 사용하는지
- target layer를 어떻게 선택하는지
- 최종 모델 architecture 변경 시 정상 동작하는지
- 결과를 이미지 파일로 저장하는지
- numpy/PIL/base64 형태로 반환 가능한지

### 설정 관련

아래와 같은 값이 여러 곳에 중복 하드코딩되어 있다면 한 곳으로 정리한다.

```text
image_size
class_names
num_classes
model_name
checkpoint_path
normalization mean/std
```

단, 서비스 구현을 위해 전체 설정 시스템을 새로 설계하지는 않는다.

---

# 5. Phase 2 — 최종 모델 준비

서비스는 모델 비교가 끝난 후 선정된 **최종 best model 하나**만 사용한다.

권장 위치:

```text
outputs/models/best_model.pth
```

현재 다른 경로를 사용한다면 기존 경로를 유지해도 된다.

## 서비스 시작 전 검증

아래가 정상 동작해야 한다.

```text
python src/pipeline/predict.py <sample-image>
```

또는 현재 프로젝트가 제공하는 기존 inference command.

최소 출력:

```text
Top-1 class
confidence
Top-3 predictions
```

서비스 구현 전에 CLI inference가 정상 동작하지 않으면 먼저 해당 문제를 수정한다.

---

# 6. Phase 3 — FastAPI inference 서버 구현

## 목표

API는 우선 아래 하나면 충분하다.

```http
POST /predict
```

선택적으로 상태 확인용:

```http
GET /health
```

를 추가한다.

---

# 7. FastAPI 서버 구조

## `service/api/main.py`

역할:

```text
FastAPI app 생성
CORS 설정
모델 inference service 초기화
/predict endpoint
/health endpoint
예외 처리
```

### 중요한 구현 규칙

다음처럼 요청마다 모델을 로드하지 않는다.

```python
@app.post("/predict")
async def predict(...):
    model = load_model(...)
```

대신 server lifecycle 또는 module initialization을 이용하여 **모델을 서버 시작 시 1회 로드**한다.

개념:

```text
FastAPI 시작
    ↓
checkpoint load 1회
    ↓
model.eval()
    ↓
요청 대기

사용자 요청
    ↓
전처리
    ↓
기존 model 재사용
    ↓
응답
```

---

# 8. `service/api/inference.py`

이 파일은 API와 PyTorch inference 로직 사이를 연결한다.

가능하면 `src/pipeline/predict.py` 내부에 이미 존재하는 함수들을 import해서 사용한다.

예시 책임:

```text
load_service_model()
preprocess_image()
predict_topk()
generate_gradcam()  # 선택
```

그러나 같은 코드가 중복된다면 `predict.py`의 공통 추론 함수를 재사용 가능한 형태로 최소 리팩터링한다.

예:

```python
def load_model(...):
    ...

def preprocess_image(...):
    ...

def predict_image(model, image, top_k=3):
    ...
```

CLI용 `predict.py`도 동일 함수를 호출하도록 유지하면 가장 좋다.

---

# 9. API 입력

`multipart/form-data`

field name:

```text
image
```

지원 파일:

```text
.jpg
.jpeg
.png
```

가능하면 MIME type도 확인한다.

잘못된 파일이면 400 수준의 명확한 에러를 반환한다.

---

# 10. API 응답 스키마

서비스에서 병변 정보를 함께 반환하도록 한다.

권장 응답:

```json
{
  "prediction": {
    "class": "basal_cell_carcinoma",
    "name_ko": "기저세포암",
    "name_en": "Basal Cell Carcinoma",
    "confidence": 0.913
  },
  "information": {
    "category": "악성 피부종양",
    "description": "사전에 작성된 설명",
    "features": [
      "특징 1",
      "특징 2"
    ],
    "precautions": [
      "주의사항 1",
      "주의사항 2"
    ]
  },
  "top3": [
    {
      "class": "basal_cell_carcinoma",
      "name_ko": "기저세포암",
      "probability": 0.913
    },
    {
      "class": "example_class_2",
      "name_ko": "예시 병변 2",
      "probability": 0.052
    },
    {
      "class": "example_class_3",
      "name_ko": "예시 병변 3",
      "probability": 0.021
    }
  ],
  "gradcam": null
}
```

주의:

- 위 class 이름들은 구조 예시일 뿐이다.
- 실제 반환 class는 **현재 10개 클래스 source of truth**를 사용한다.
- confidence/probability는 프론트에서 `%`로 변환하기 편하도록 0~1 float를 반환한다.

---

# 11. `schemas.py`

Pydantic model을 사용하여 응답 형태를 명확하게 정의한다.

예:

```text
PredictionResult
TopPrediction
LesionInformation
PredictResponse
```

필요 이상으로 복잡하게 쪼개지는 않는다.

---

# 12. 10개 병변 사전 정보 기능

이번 서비스에서 모델 예측 결과를 단순 class label로 끝내지 않고, 해당 label에 대응하는 **사전 정의된 병변 설명**을 함께 제공한다.

이 기능은 생성형 AI를 사용하지 않는다.

구조:

```text
모델 Top-1 class
        ↓
lesion_info.json
        ↓
해당 class 정보 조회
        ↓
API 응답
```

---

# 13. `lesion_info.json`

10개 병변 정보를 정적 JSON으로 관리한다.

구조 예시:

```json
{
  "basal_cell_carcinoma": {
    "name_ko": "기저세포암",
    "name_en": "Basal Cell Carcinoma",
    "category": "악성 피부종양",
    "description": "설명",
    "features": [
      "주요 특징"
    ],
    "precautions": [
      "주의사항"
    ]
  }
}
```

## 매우 중요한 사항

Codex는 10개 병변에 대한 의료 설명을 임의 생성하지 않는다.

현재 repository 또는 사용자가 제공한 정보에 실제 설명 데이터가 없다면:

1. 10개 class key를 현재 class list에서 가져온다.
2. 정보 데이터 구조만 만들어 둔다.
3. 내용이 없는 필드는 빈 문자열/빈 배열 또는 명확한 placeholder로 둔다.
4. README 또는 TODO에 사용자가 검증된 내용을 채워야 한다고 남긴다.

예:

```json
{
  "actual_class_name": {
    "name_ko": "",
    "name_en": "",
    "category": "",
    "description": "",
    "features": [],
    "precautions": []
  }
}
```

의료 문구를 추측하여 채우지 않는다.

---

# 14. 사용자에게 보여줄 표현 원칙

결과 화면에서 확정 진단처럼 표현하지 않는다.

피해야 할 문구:

```text
기저세포암입니다.
악성 흑색종입니다.
```

권장 표현:

```text
기저세포암으로 가장 높게 예측되었습니다.

모델 예측 결과
기저세포암 91.3%

AI 모델이 해당 이미지에서 가장 높은 확률로 예측한 유형입니다.
```

고정 안내문도 둔다.

```text
본 결과는 학습 프로젝트의 이미지 분류 모델이 제공하는 참고용 예측 결과이며,
의료적 진단을 대체하지 않습니다.
```

---

# 15. Grad-CAM 연결

Grad-CAM은 **선택 기능**이다.

우선순위:

```text
1. Top-3 Prediction
2. 병변 정보
3. 안정적인 API
4. Next.js 연결
5. Grad-CAM
```

기존 `gradcam.py`가 최종 모델에서 정상 동작하고 구현 비용이 크지 않은 경우에만 서비스에 연결한다.

---

# 16. Grad-CAM 응답 방식

가능하면 다음 두 방법 중 단순한 쪽을 선택한다.

## 방법 A — base64

API 응답:

```json
{
  "gradcam": {
    "image_base64": "..."
  }
}
```

장점:

- 별도 파일 serving 불필요
- 프로젝트 데모에 간단함

단점:

- 응답 크기 증가

## 방법 B — 별도 endpoint

예:

```http
POST /predict
GET /gradcam/{id}
```

이번 프로젝트 규모에서는 구조가 복잡해질 수 있으므로 **A를 우선 고려**한다.

Grad-CAM 구현 때문에 endpoint, 저장소, ID 관리가 복잡해지면 Grad-CAM 서비스 연동은 생략한다.

---

# 17. Phase 4 — Next.js 웹앱 구현

서비스는 **웹사이트 + 웹앱 UX** 형태로 구현한다.

페이지를 과도하게 늘리지 않는다.

권장:

```text
/           이미지 분석
/lesions    10개 병변 정보
```

`/about`은 필요할 때만 추가한다.

로그인/회원가입/대시보드/내 분석 기록/설정 페이지는 만들지 않는다.

---

# 18. 메인 페이지 `/`

한 화면에서 다음 흐름을 제공한다.

```text
Hero
 ↓
이미지 Upload
 ↓
Preview
 ↓
분석하기
 ↓
Loading
 ↓
Result
```

페이지 이동 없이 결과가 같은 화면에 나타나는 웹앱 UX를 우선한다.

---

# 19. 메인 화면 구성

권장 순서:

```text
1. Header
2. Hero
3. Upload 영역
4. Image Preview
5. 분석하기 Button
6. Loading UI
7. 결과 Card
8. Top-3
9. 예측 병변 설명
10. 주요 특징 / 주의사항
11. Grad-CAM (선택)
12. 참고용 안내 문구
13. 분석 가능한 10개 병변
```

---

# 20. 업로드 UX

지원:

- Click file upload
- 가능하면 Drag & Drop
- JPG/JPEG/PNG

업로드 후:

```text
파일명
이미지 preview
다른 이미지 선택
분석하기
```

분석 중에는 버튼 중복 클릭을 방지한다.

---

# 21. 결과 UI

가장 중요한 정보를 먼저 보여준다.

예:

```text
분석 결과

기저세포암으로 가장 높게 예측되었습니다.
Basal Cell Carcinoma

91.3%

Top-3
1. 기저세포암        91.3%
2. 보웬병             5.2%
3. 편평세포암         2.1%
```

그 아래:

```text
병변 설명
주요 특징
주의사항
Grad-CAM
참고용 안내 문구
```

Top-1 결과보다 Grad-CAM을 더 크게 강조하지 않는다.

---

# 22. `/lesions`

현재 실제 10개 class에 대한 카드 목록을 제공한다.

```text
10개 피부 병변

[ 병변 1 ]
[ 병변 2 ]
[ 병변 3 ]
...
```

각 카드 선택 시 상세 설명을 보여준다.

구현은 다음 중 간단한 방식을 사용한다.

- Accordion
- Modal
- Expandable Card

별도의 `/lesions/[id]` 동적 페이지는 꼭 필요하지 않다.

---

# 23. 프론트엔드 데이터 관리

병변 정보를 API에서 제공한다면 프론트엔드에 동일한 데이터를 중복 하드코딩하지 않는다.

권장 방법:

### 메인 분석 결과

`POST /predict`의 `information`을 그대로 사용.

### `/lesions`

두 가지 중 하나:

1. 별도 `GET /lesions` endpoint 추가
2. `lesion_info.json` 기반의 프론트 정적 데이터 생성

가능하면 데이터 source가 하나가 되도록 한다.

서비스 구조가 복잡해지지 않는 범위에서는 `GET /lesions`가 깔끔하다.

선택 endpoint:

```http
GET /lesions
```

응답:

```json
{
  "lesions": [...]
}
```

---

# 24. API client

프론트에서 FastAPI 호출 코드를 component 내부에 흩어놓지 않는다.

예:

```text
service/web/lib/api.ts
```

역할:

```text
predictImage(file)
getLesions()
```

환경변수:

```text
NEXT_PUBLIC_API_BASE_URL
```

로 API 주소를 관리한다.

예:

```text
http://localhost:8000
```

주소를 component에 직접 하드코딩하지 않는다.

---

# 25. CORS

개발 환경 예:

```text
Next.js: http://localhost:3000
FastAPI: http://localhost:8000
```

FastAPI CORS 설정에 프론트 URL을 명시한다.

개발 편의를 위해 무조건 `*`를 사용하는 것보다 환경변수 기반 origin 설정을 우선한다.

단, 프로젝트 복잡도가 크게 증가하면 localhost 기준 최소 설정으로 구현한다.

---

# 26. 에러 처리

다음 상황에 대한 UI/API 처리를 한다.

| 상황 | 처리 |
| --- | --- |
| 이미지 미선택 | 분석 버튼 비활성화 또는 안내 |
| JPG | 정상 |
| JPEG | 정상 |
| PNG | 정상 |
| 이미지 아닌 파일 | 400 + 사용자 안내 |
| 깨진 이미지 | 400 수준 처리 |
| API 연결 실패 | 프론트 오류 메시지 |
| checkpoint 없음 | 서버 시작 단계에서 명확한 에러 |
| class mapping 불일치 | 서버 시작 또는 테스트에서 검증 |
| inference 실패 | 500 + 내부 로그 |
| Grad-CAM 실패 | 예측 결과는 유지하고 Grad-CAM만 숨김 가능 |

---

# 27. 모델 클래스 매핑 검증

이 프로젝트에서 가장 위험한 오류 중 하나는 **학습 시 class index 순서와 서비스 class name 순서가 달라지는 것**이다.

예:

```text
학습:
0 = actinic_keratosis
1 = basal_cell_carcinoma

서비스:
0 = basal_cell_carcinoma
1 = actinic_keratosis
```

이런 오류가 절대 발생하지 않게 한다.

## 권장

checkpoint에 class mapping을 함께 저장하고 있다면 그것을 사용한다.

예:

```python
{
    "model_state_dict": ...,
    "class_names": [...],
    "config": ...
}
```

현재 checkpoint가 그렇지 않다면 기존 dataset/config에서 사용하는 동일한 class list를 가져온다.

가능하면 alphabetic sort를 서비스에서 새로 수행하지 않는다.

---

# 28. 모델 전처리 일치 검증

학습/평가/서비스에서 아래가 동일해야 한다.

```text
image_size
RGB conversion
ToTensor
Normalize mean
Normalize std
```

Validation/Test inference에는 학습용 random augmentation을 적용하지 않는다.

예:

```text
RandomHorizontalFlip
RandomRotation
ColorJitter
```

같은 transform이 서비스 inference에 들어가면 안 된다.

---

# 29. Device 처리

서비스는 CPU에서도 실행 가능해야 한다.

권장:

```text
CUDA 사용 가능 → cuda
Apple MPS 사용 가능 → 필요 시 mps
그 외 → cpu
```

단, checkpoint loading 시 `map_location`을 적절히 사용하여 GPU에서 저장한 모델을 CPU 서버에서도 로드할 수 있게 한다.

서비스의 1차 목표는 GPU 최적화가 아니라 **재현 가능한 inference**다.

---

# 30. Docker 적용 범위

이번 프로젝트에서는 **학습 환경 전체를 Docker화하지 않는다.**

권장 구조:

```text
모델 학습
.venv
+
각 팀원 GPU / MPS / CPU 환경

            ↓

최종 모델

            ↓

FastAPI inference service
Docker
```

Docker를 쓰는 이유는 최종 inference 서버의 실행 환경을 통일하기 위함이다.

---

# 31. Docker 우선 구현 대상

1차 Docker 대상:

```text
FastAPI + PyTorch inference
```

Next.js까지 반드시 같은 container로 묶을 필요는 없다.

시간이 충분한 경우에만 `docker-compose.yml`을 사용하여:

```text
web
api
```

두 서비스를 함께 실행하도록 구성한다.

---

# 32. Dockerfile 요구사항

FastAPI inference 서버 Dockerfile은 대략 다음 환경을 재현해야 한다.

```text
Python
PyTorch
TorchVision
FastAPI
Uvicorn
Pillow
프로젝트에서 inference에 실제 필요한 dependencies
```

학습에만 필요한 패키지를 무조건 전부 포함할 필요는 없다.

단, 기존 `requirements.txt` 분리가 오히려 시간을 많이 사용한다면 우선 기존 requirements를 사용해도 된다.

---

# 33. Docker에서 모델 checkpoint

checkpoint를 image에 COPY할지 volume mount할지 선택해야 한다.

이번 프로젝트 데모 목적에서는 단순성을 우선한다.

예:

```dockerfile
COPY outputs/models/best_model.pth /app/outputs/models/best_model.pth
```

단, checkpoint 파일이 Git에 포함되지 않거나 용량이 너무 크다면 README에 배치 방법을 명시한다.

---

# 34. DB 사용 금지

현재 서비스는 다음 구조다.

```text
이미지 입력
→ 추론
→ 결과 반환
```

stateless 서비스이므로 DB를 추가하지 않는다.

추가하지 말 것:

```text
PostgreSQL
Supabase
MongoDB
SQLite
회원 테이블
분석 기록 테이블
```

향후 분석 이력 저장 기능이 생기면 그때 DB를 고려한다.

---

# 35. 구현하지 않을 기능

이번 작업 범위에서 제외:

- 로그인
- 회원가입
- OAuth
- 사용자 프로필
- 분석 기록 저장
- 관리자 페이지
- CMS
- 커뮤니티
- 게시판
- AI 챗봇
- RAG
- LLM
- 결제
- 푸시 알림
- 복잡한 클라우드 인프라
- Kubernetes
- Redis
- Celery
- 메시지 큐
- Microservice 분리

Codex가 임의로 scope를 확장하지 않는다.

---

# 36. 권장 구현 순서

아래 순서를 반드시 우선한다.

## Step 1

현재 repository 분석

```text
predict.py
evaluate.py
gradcam.py
model
config
checkpoint
class mapping
```

## Step 2

10개 클래스 기준 검증

```text
num_classes
class_names
checkpoint output
```

## Step 3

기존 CLI inference 정상 동작 확인

## Step 4

inference 공통 함수 정리

## Step 5

FastAPI `/health`

## Step 6

FastAPI `/predict`

## Step 7

Swagger에서 이미지 업로드 테스트

## Step 8

병변 정보 JSON 연결

## Step 9

Next.js upload UI

## Step 10

Next.js → FastAPI 연결

## Step 11

Top-3 + 병변 정보 표시

## Step 12

에러 처리

## Step 13

Grad-CAM 연결 검토

## Step 14

Docker

## Step 15

README 실행 방법 갱신

---

# 37. `/health`

간단한 health endpoint를 구현한다.

예:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "efficientnet_b0",
  "num_classes": 10
}
```

`model_name`은 실제 최종 모델을 사용한다.

임의로 EfficientNet으로 하드코딩하지 않는다.

---

# 38. 성능 관련 최소 요구사항

### 모델 로딩

요청마다 재로딩 금지.

### Gradient

inference:

```python
torch.inference_mode()
```

또는:

```python
torch.no_grad()
```

사용.

가능하면 `torch.inference_mode()`를 우선한다.

### model mode

```python
model.eval()
```

필수.

---

# 39. 추론 시간 측정

가능하면 inference latency를 측정할 수 있게 내부적으로 구현한다.

API 응답에 선택적으로:

```json
{
  "inference_time_ms": 42.7
}
```

를 포함해도 된다.

포트폴리오의 모델 비교에서 추론 속도를 보여줄 수 있기 때문이다.

단, 전체 HTTP round-trip과 순수 모델 inference 시간을 혼동하지 않도록 변수명을 명확히 한다.

---

# 40. 확률 표현

API:

```text
0.913
```

프론트:

```text
91.3%
```

Top-3 probability를 합쳤을 때 반드시 100%가 되어야 하는 것은 아니다.

Softmax 전체 10개 class 중 상위 3개만 보여주기 때문이다.

---

# 41. confidence 해석 주의

UI에서 Softmax probability를 실제 의학적 확률처럼 표현하지 않는다.

피해야 할 표현:

```text
기저세포암일 확률 91.3%
```

권장:

```text
모델 예측 점수 91.3%

분류 모델이 가장 높은 점수를 부여한 유형
```

또는:

```text
예측 신뢰도 91.3%
```

단, README에서는 softmax output이라는 점을 명시한다.

---

# 42. UI 디자인 방향

서비스명:

```text
skina
```

전체 디자인:

- 심플
- 의료 서비스처럼 과도하게 무겁지 않음
- 흰 배경 중심
- 카드 기반
- 충분한 여백
- 모바일에서도 깨지지 않는 responsive layout

핵심 CTA:

```text
이미지 선택
분석하기
```

불필요하게 화려한 animation은 넣지 않는다.

---

# 43. 결과 화면 우선순위

```text
★★★★★ Top-1 예측
★★★★★ Top-3
★★★★★ 병변 사전 정보
★★★★☆ 업로드 이미지
★★★☆☆ Grad-CAM
```

Grad-CAM이 없는 상태에서도 서비스가 완성되어야 한다.

---

# 44. 기본 서비스 플로우

```text
┌──────────────────────────────┐
│            skina             │
│                              │
│      피부 이미지 업로드      │
│                              │
│       [ 이미지 선택 ]        │
│                              │
│         [ 분석하기 ]         │
└──────────────────────────────┘

              ↓

┌──────────────────────────────┐
│ 분석 결과                    │
│                              │
│ 가장 높게 예측된 유형        │
│ 기저세포암                   │
│ Basal Cell Carcinoma         │
│ 모델 예측 점수 91.3%         │
│                              │
│ Top-3                        │
│ 1. ...                       │
│ 2. ...                       │
│ 3. ...                       │
│                              │
│ 병변 설명                    │
│ 주요 특징                    │
│ 주의사항                     │
│                              │
│ [Grad-CAM 선택]              │
│                              │
│ 참고용 예측 결과 안내        │
└──────────────────────────────┘
```

위 병변명은 화면 구조 예시이며 실제 클래스와 데이터에 맞춰 렌더링한다.

---

# 45. 테스트

최소 수동/자동 테스트 항목:

```text
[ ] GET /health 정상
[ ] 서버 시작 시 모델 1회 로드
[ ] JPG 업로드
[ ] JPEG 업로드
[ ] PNG 업로드
[ ] 이미지 아닌 파일 거부
[ ] 손상 이미지 처리
[ ] 큰 이미지 resize 후 inference
[ ] Top-1 정상
[ ] Top-3 정상
[ ] class index mapping 정상
[ ] 10개 클래스 기준 정상
[ ] lesion info mapping 정상
[ ] API 서버 down 시 frontend 오류 처리
[ ] checkpoint 없음 시 명확한 서버 오류
[ ] CPU inference 정상
[ ] Next.js image preview
[ ] loading state
[ ] 중복 submit 방지
[ ] responsive UI
[ ] Grad-CAM 선택 기능 정상
```

---

# 46. 코드 작성 시 요구사항

- type hint 적극 사용
- 함수가 지나치게 길어지지 않게 분리
- 중복 inference 코드 최소화
- path 하드코딩 최소화
- 가능한 경로는 `pathlib.Path` 사용
- 모델/클래스/config 불일치 시 조용히 실패하지 말고 명확한 에러 발생
- debug print 남발하지 않기
- 필요한 로그만 사용
- `.env.example` 제공
- 실제 `.env` commit 금지
- 기존 코드 포맷과 스타일 유지

---

# 47. 환경변수 예시

Backend:

```env
MODEL_PATH=outputs/models/best_model.pth
```

필요하면:

```env
ALLOWED_ORIGINS=http://localhost:3000
```

Frontend:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

# 48. README에 추가할 실행 방법

최종적으로 README에 최소 다음을 작성한다.

## Backend without Docker

Windows 예:

```bash
.venv\Scripts\activate
uvicorn service.api.main:app --reload
```

macOS/Linux 예:

```bash
source .venv/bin/activate
uvicorn service.api.main:app --reload
```

## Frontend

```bash
cd service/web
npm install
npm run dev
```

## 접속

```text
Frontend
http://localhost:3000

FastAPI Swagger
http://localhost:8000/docs
```

## Docker

실제 구현 방식에 맞게 명령을 작성한다.

예:

```bash
docker build -t skina-api .
docker run -p 8000:8000 skina-api
```

또는 compose를 구현한 경우:

```bash
docker compose up --build
```

---

# 49. README에 아키텍처 추가

```text
User
 ↓
Next.js
 ↓ HTTP multipart/form-data
FastAPI
 ↓
Inference Service
 ↓
PyTorch Final Model
 ↓
Softmax
 ├─ Top-3
 ├─ Lesion Metadata
 └─ Grad-CAM (optional)
 ↓
Next.js Result UI
```

---

# 50. 완료 조건 Definition of Done

다음 조건을 만족하면 서비스 구현을 완료한 것으로 본다.

## 필수

- [ ] 전체 서비스가 현재 **10개 클래스** 기준으로 동작한다.
- [ ] 기존 inference 로직을 최대한 재사용한다.
- [ ] 최종 checkpoint를 FastAPI가 정상 로드한다.
- [ ] 서버 시작 시 모델을 1회만 로드한다.
- [ ] `POST /predict`가 이미지 파일을 받는다.
- [ ] Top-1 결과를 반환한다.
- [ ] Top-3 결과를 반환한다.
- [ ] 각 class에 대응하는 병변 정보 구조가 존재한다.
- [ ] Next.js에서 이미지 선택 및 preview가 된다.
- [ ] Next.js에서 FastAPI로 이미지를 전송한다.
- [ ] 결과 화면에 Top-1 / Top-3가 표시된다.
- [ ] 결과 화면에 병변 설명 영역이 표시된다.
- [ ] 잘못된 파일/서버 오류를 사용자에게 표시한다.
- [ ] README에 실행 방법이 정리되어 있다.

## 선택

- [ ] Grad-CAM이 결과 화면에 표시된다.
- [ ] `/lesions` 페이지가 구현된다.
- [ ] `GET /lesions`가 구현된다.
- [ ] inference time이 응답에 포함된다.
- [ ] FastAPI inference server가 Docker로 실행된다.
- [ ] docker-compose로 frontend/backend를 함께 실행한다.

---

# 51. 구현 결과 보고 방식

작업 완료 후 Codex는 다음 형식으로 요약한다.

```text
1. 구현한 내용
2. 새로 생성한 파일
3. 수정한 파일
4. 15 → 10 클래스 변경을 반영한 위치
5. 모델 로딩 방식
6. API endpoint
7. Frontend 동작
8. Grad-CAM 적용 여부
9. Docker 적용 여부
10. 실행 명령
11. 직접 확인한 테스트
12. 남은 TODO
```

---

# 52. Codex에게 최종 지시

위 요구사항을 기준으로 현재 repository를 먼저 분석한 뒤 구현하라.

중요:

1. 현재 코드를 무시하고 새 프로젝트를 만들지 말 것.
2. 기존 `predict.py`, `evaluate.py`, `gradcam.py`, 모델/config 구조를 먼저 확인할 것.
3. 현재 데이터셋의 실제 10개 class 목록과 순서를 확인할 것.
4. 기존 15-class 하드코딩을 모두 찾아 10-class 구조와 충돌하지 않는지 검증할 것.
5. 학습과 inference transform이 일치하는지 확인할 것.
6. checkpoint와 class mapping이 일치하는지 확인할 것.
7. FastAPI inference를 먼저 완성하고 Swagger에서 테스트할 것.
8. 그 다음 Next.js를 연결할 것.
9. Grad-CAM은 필수 기능 완성 후 연결할 것.
10. Docker는 마지막 단계에서 적용할 것.
11. DB, 로그인, 챗봇 등 범위 밖 기능을 추가하지 말 것.
12. 의료 병변 설명은 repository에 검증된 내용이 없는 경우 임의 생성하지 말 것.
13. 구현 중 기존 기능을 깨뜨리지 말 것.
14. 가능한 경우 수정 후 실제 명령을 실행하여 동작을 검증할 것.
15. 작업 마지막에 변경 사항, 테스트 결과, 실행 방법을 요약할 것.

최종 목표는 다음 한 줄이다.

```text
Image → Next.js → FastAPI → PyTorch Final Model → Top-3 + Lesion Info (+ Grad-CAM) → Result UI
```

복잡한 서비스보다 **안정적으로 시연 가능한 End-to-End AI 서비스**를 완성하는 것을 최우선으로 한다.
