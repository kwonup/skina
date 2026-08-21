# skina 서비스 구현 로드맵

이 문서는 `docs/skina_service_plan.md`의 요구사항과 현재 저장소 상태를 함께
검토하여 작성한 구현 순서이다. 목표는 기존 학습 코드를 유지하면서 다음 흐름을
작은 커밋 단위로 완성하는 것이다.

```text
이미지 업로드 -> Next.js -> FastAPI -> PyTorch -> Top-3 + 병변 정보
                                             \-> Grad-CAM (선택)
```

## 1. 현재 저장소 기준선

### 확인된 내용

- 현재 서비스 대상 데이터는 `data/processed/{train,val,test}`의 10개 클래스이다.
- 세 split의 클래스 이름과 알파벳순 index가 서로 같다.
- 현재 10개 클래스는 다음 순서이다.

```text
0  actinic_keratosis
1  basal_cell_carcinoma
2  dermatofibroma
3  hemangioma
4  lentigo
5  malignant_melanoma
6  melanocytic_nevus
7  seborrheic_keratosis
8  squamous_cell_carcinoma
9  wart
```

- `outputs/models/efficientnet_b0_best.pth`는 위와 동일한 10개 `class_names`를
  포함한다.
- 해당 checkpoint의 모델은 `efficientnet_b0`, 입력 크기는 224이며 classifier
  출력도 10개 클래스이다.
- 기존 CLI 추론은 현재 checkpoint로 정상 동작한다.
- 평가와 추론은 checkpoint의 `class_names`를 읽어 모델 출력 크기를 정하므로
  기본 구조는 재사용할 수 있다.
- 검증/추론 transform은 `Resize(224, 224) -> ToTensor -> ImageNet Normalize`로
  공통화되어 있다.
- Grad-CAM은 외부 전용 라이브러리 없이 현재 코드에서 직접 계산하며
  EfficientNet-B0의 마지막 feature block을 지원한다.

### 먼저 해결해야 할 불일치

- README, `src/data/prepare_data.py`, 네 모델 생성 함수의 기본값에는 15개 클래스
  기준이 남아 있다.
- `data/raw`는 15개 클래스지만 `data/processed`는 10개 클래스이다.
- 현재 processed 데이터는 클래스마다 train 1,000장, val 100장, test 100장인데,
  기존 `prepare_data.py`는 클래스마다 800/50/50을 만드는 코드이다. 따라서 현재
  코드만으로 최종 학습 데이터를 재현할 수 없다.
- 10클래스 checkpoint만으로는 최종 모델 선정 절차를 증명할 평가 결과 파일이
  없으므로, 서비스 모델을 EfficientNet-B0로 확정했다는 팀 합의가 필요하다.
- checkpoint는 `.gitignore` 대상이므로 Docker와 다른 개발 환경에서 전달할
  방법을 정해야 한다.
- 검증된 병변 설명 데이터가 저장소에 없으므로 의료 설명은 placeholder로
  구현하고, 검수된 문구를 별도로 받아야 한다.

## 2. 구현 전 결정 사항

아래 세 항목은 코드로 추측하지 않는다.

1. `efficientnet_b0_best.pth`를 서비스 최종 모델로 확정한다.
2. 현재 10클래스 processed 데이터의 생성 원본과 1,000/100/100 split 규칙을
   기록한다. 확인 전에는 `prepare_data.py --overwrite`를 실행하지 않는다.
3. 병변별 한글명·영문명·설명·특징·주의사항은 검수된 자료를 받기 전까지 빈 값과
   빈 배열로 둔다.

서비스 구현 자체는 1번을 현재 저장소 기준으로 가정하면 진행할 수 있다. 2번은
학습 재현성을 위한 별도 완료 조건이고, API 추론을 막지는 않는다.

## 3. 커밋 단위 구현 계획

각 커밋은 코드와 해당 범위의 테스트를 함께 포함한다. 이후 커밋이 아직 없어도
해당 커밋 시점의 기존 CLI 또는 새 기능이 정상 동작해야 한다.

### 커밋 1. 10개 클래스 계약 확정

추천 커밋명:

```text
refactor: 10개 클래스 기준과 매핑 검증을 통합
```

주요 작업:

- `configs/class_names.json`을 만들고 현재 10개 클래스와 순서를 기록한다.
- 학습 데이터, checkpoint, 병변 정보가 이 목록과 정확히 일치하는지 검사하는
  공통 loader/validator를 추가한다.
- 모델 생성 함수의 `num_classes=15` 기본값을 제거하고 호출자가 클래스 수를
  명시하게 한다. 숫자 10을 모델 파일마다 다시 하드코딩하지 않는다.
- `train.py`, `evaluate.py`, `predict.py`, `gradcam.py`에서 클래스 수·순서·모델
  출력 차원의 불일치를 조기에 오류로 처리한다.
- `prepare_data.py`의 15클래스 상수는 공통 목록을 사용하게 바꾸되, 새 데이터
  원본과 split 규칙이 확인되기 전에는 현재 processed 데이터를 재생성하지
  못하도록 명확히 안내한다.
- README의 15클래스 표현과 현재 데이터 수치는 우선 사실에 맞게 바로잡는다.

검증 기준:

- processed 세 split, 중앙 클래스 목록, 10클래스 checkpoint의 순서가 같다.
- 15클래스 이전 checkpoint는 명확한 mapping 오류로 거부된다.
- `rg` 검색에서 서비스 실행 경로의 `num_classes=15` 하드코딩이 사라진다.

### 커밋 2. 공통 추론 로직 분리

추천 커밋명:

```text
refactor: 모델 로딩과 이미지 추론 로직을 공통화
```

주요 작업:

- `predict.py`의 모델 로딩, checkpoint 검증, PIL 전처리, Softmax, Top-K 계산을
  재사용 가능한 함수 또는 작은 inference 객체로 분리한다.
- 로드된 모델, device, class names, image size, model name을 하나의 객체로
  관리한다.
- 공통 추론 함수는 파일 경로뿐 아니라 이미 열린 PIL 이미지도 입력받게 한다.
- 모델은 `eval()` 상태로 두고 일반 예측은 `torch.inference_mode()`에서 실행한다.
- CLI `predict.py`는 같은 공통 함수를 호출하는 얇은 wrapper로 유지한다.
- 출력용 `print`와 순수 추론 결과 생성을 분리한다.

검증 기준:

- 기존 CLI 명령이 계속 Top-3를 출력한다.
- 같은 이미지에 대해 리팩터링 전후 Top-3 class와 점수가 허용 오차 내에서 같다.
- CPU `map_location`으로도 checkpoint를 로드할 수 있다.
- 모델 로딩 횟수를 검사할 수 있는 단위 테스트가 있다.

### 커밋 3. FastAPI 생명주기와 health API 추가

추천 커밋명:

```text
feat: FastAPI 서버 생명주기와 상태 확인 API 추가
```

주요 작업:

- `service/api` Python package와 `main.py`, 설정 모듈을 만든다.
- FastAPI, Uvicorn, multipart 처리와 테스트에 필요한 backend 의존성을 추가한다.
- lifespan에서 공통 inference 객체를 서버 시작 시 한 번만 생성하고 app state에
  보관한다.
- `MODEL_PATH`, `ALLOWED_ORIGINS` 환경변수와 backend `.env.example`을 제공한다.
- localhost Next.js origin만 기본 허용하고 CORS를 환경변수로 확장한다.
- `GET /health`에서 model loaded, model name, num classes, device 정보를 반환한다.
- checkpoint 없음, 모델명 불일치, 클래스 mapping 불일치는 시작 단계에서
  원인이 드러나는 오류로 처리한다.

검증 기준:

- `/health`가 200을 반환하고 `num_classes`가 10이다.
- 여러 health 요청에도 checkpoint는 한 번만 로드된다.
- 잘못된 `MODEL_PATH`로 시작하면 원인을 포함한 오류가 발생한다.

### 커밋 4. 이미지 Top-3 예측 API 구현

추천 커밋명:

```text
feat: 이미지 검증과 Top-3 예측 API 구현
```

주요 작업:

- Pydantic으로 Top prediction, prediction summary, predict response schema를
  정의한다.
- `POST /predict`에서 `multipart/form-data`의 `image` 필드를 받는다.
- JPG/JPEG/PNG MIME type, 확장자, 빈 파일, 최대 업로드 크기와 실제 이미지
  decode 가능 여부를 검사한다.
- RGB 변환 후 커밋 2의 공통 추론 함수를 호출한다.
- Top-1, Top-3, 0~1 probability, 순수 모델 추론 시간(ms)을 반환한다.
- 동기 PyTorch 작업은 event loop를 막지 않도록 threadpool에서 실행한다.
- 예상 가능한 입력 오류는 400 계열, 내부 추론 오류는 로그를 남기고 500으로
  구분한다.

검증 기준:

- 정상 JPG/JPEG/PNG가 200과 Top-3를 반환한다.
- 텍스트 파일, 위조 MIME, 손상 이미지, 빈 파일은 400 계열로 거부된다.
- Top-3는 내림차순이고 모든 probability는 0~1 범위이다.
- 반복 요청에도 동일한 app state의 모델을 사용한다.

### 커밋 5. 병변 정보와 lesions API 연결

추천 커밋명:

```text
feat: 병변 정보 조회와 예측 응답 연동
```

주요 작업:

- `service/api/lesion_info.json`에 정확히 10개 class key를 만든다.
- 검증 자료가 없는 `name_ko`, `name_en`, `category`, `description`은 빈 문자열,
  `features`, `precautions`는 빈 배열로 둔다.
- 시작 시 병변 정보 key와 class mapping이 정확히 일치하는지 검사한다.
- `/predict`의 Top-1에 해당하는 information을 합쳐 반환한다.
- `GET /lesions`를 추가하여 웹이 동일한 backend 데이터를 사용하게 한다.
- 프론트가 반드시 표시할 고정 의료 면책 문구는 상수로 관리한다.

검증 기준:

- 누락되거나 추가된 lesion key가 있으면 서버 시작 시 실패한다.
- `/predict`의 Top-1 class와 information의 class가 같다.
- `/lesions`가 checkpoint 순서대로 정확히 10개를 반환한다.
- 검증되지 않은 의료 설명이 코드에 임의로 추가되지 않는다.

### 커밋 6. Next.js 웹앱 기본 구조 생성

추천 커밋명:

```text
feat: Next.js 웹앱 기본 화면과 공통 레이아웃 구성
```

주요 작업:

- `service/web`에 TypeScript, App Router 기반 Next.js 앱을 만든다.
- Header, Hero, 메인 container, 공통 카드와 의료 면책 영역을 구성한다.
- 흰 배경, 충분한 여백, 단순한 카드 중심의 responsive 스타일을 적용한다.
- `NEXT_PUBLIC_API_BASE_URL`을 `.env.example`로 제공한다.
- API 응답 TypeScript type과 `lib/api.ts`의 client 골격을 만든다.
- 로그인, DB, 대시보드 등 범위 밖 페이지는 만들지 않는다.

검증 기준:

- `npm run lint`와 `npm run build`가 성공한다.
- 320px 모바일 폭과 데스크톱에서 가로 overflow가 없다.
- API URL이 component에 하드코딩되어 있지 않다.

### 커밋 7. 이미지 업로드와 API 연동

추천 커밋명:

```text
feat: 이미지 업로드와 FastAPI 연동 구현
```

주요 작업:

- 클릭 선택과 drag & drop을 지원하는 upload component를 만든다.
- 선택 파일명, 미리보기, 다시 선택, 분석 버튼을 제공한다.
- 브라우저에서도 JPG/JPEG/PNG와 파일 크기를 사전 검사한다.
- `predictImage(file)`이 FormData의 `image` 필드로 `/predict`를 호출하게 한다.
- 로딩 중 중복 submit을 차단하고 loading indicator를 표시한다.
- 새 파일 선택과 component 해제 시 preview object URL을 해제한다.
- API 연결 실패, 4xx 응답, 서버 오류를 사용자가 이해할 수 있는 한국어로
  표시한다.

검증 기준:

- 이미지 선택 즉시 preview가 보인다.
- 분석 요청의 field name과 content type이 backend 계약과 일치한다.
- 로딩 중 버튼이 비활성화되고 중복 요청이 발생하지 않는다.
- 서버가 꺼져 있어도 페이지가 깨지지 않고 오류 안내가 보인다.

### 커밋 8. 예측 결과 화면 구현

추천 커밋명:

```text
feat: Top-3 예측과 병변 정보 결과 화면 구현
```

주요 작업:

- Top-1 class, 표시 이름, 모델 예측 점수와 영문명을 가장 먼저 보여준다.
- Top-3를 순위와 progress bar로 표시하고 0~1 값을 화면에서 퍼센트로 바꾼다.
- 병변 설명, 주요 특징, 주의사항을 조건부 렌더링한다.
- 빈 placeholder는 허위 문구로 채우지 않고 `검수된 정보 준비 중`처럼 데이터가
  없음을 명시한다.
- `~입니다` 같은 확정 진단 표현을 쓰지 않고 `가장 높게 예측되었습니다`를 쓴다.
- 의료 진단을 대체하지 않는다는 고정 안내를 항상 노출한다.
- 다른 이미지를 선택하면 이전 결과와 오류 상태를 초기화한다.

검증 기준:

- Top-1과 Top-3 첫 항목이 일치한다.
- 점수가 소프트맥스 기반 모델 출력임을 UI 또는 도움말에서 알 수 있다.
- 긴 class 이름과 빈 병변 정보에서도 모바일 레이아웃이 깨지지 않는다.

### 커밋 9. 10개 병변 목록 화면 구현

추천 커밋명:

```text
feat: 10개 병변 정보 목록 화면 추가
```

주요 작업:

- `/lesions`에서 `GET /lesions` 결과를 불러온다.
- 10개 병변을 카드와 accordion 방식으로 표시한다.
- 로딩, 빈 데이터, API 오류 상태를 처리한다.
- Header에서 `/`와 `/lesions`로 이동할 수 있게 한다.
- frontend에 별도 병변 JSON을 복사하지 않는다.

검증 기준:

- backend 순서대로 10개 카드가 보인다.
- 정보를 펼치고 닫을 수 있으며 키보드로도 조작할 수 있다.
- API 장애 시 빈 화면 대신 오류 안내가 보인다.

### 커밋 10. Grad-CAM 선택 연동

추천 커밋명:

```text
feat: 선택적 Grad-CAM 시각화 연동
```

착수 조건:

- 커밋 1~9의 필수 흐름이 안정적으로 동작한다.
- 최종 EfficientNet-B0 checkpoint로 기존 Grad-CAM CLI 결과를 육안 검증했다.

주요 작업:

- 기존 Grad-CAM 계산과 overlay 생성을 CLI 출력 파일뿐 아니라 메모리의 PNG로도
  반환할 수 있게 최소 리팩터링한다.
- `ENABLE_GRADCAM` 환경변수로 기능을 켜고 끈다.
- 단순한 데모 구조를 위해 PNG base64를 `/predict` 응답에 선택적으로 포함한다.
- Grad-CAM은 gradient가 필요하므로 일반 예측의 `inference_mode()`와 실행 경로를
  분리한다.
- 공유 모델에 임시 hook과 backward를 사용하는 동안 동시 요청이 섞이지 않도록
  lock 또는 별도 직렬화 구간을 둔다.
- Grad-CAM 실패 시 Top-3 응답은 유지하고 `gradcam`만 null로 둔다.
- 웹에서는 Top-1 결과보다 작게, 조건부 보조 시각화로 표시한다.

검증 기준:

- 기능을 끄면 기존 응답과 성능에 영향이 없다.
- 기능을 켜면 decode 가능한 PNG base64가 반환된다.
- Grad-CAM 실패를 강제해도 Top-3 결과는 정상 표시된다.
- 연속 요청 후 forward hook이 누적되지 않는다.

### 커밋 11. FastAPI Docker 실행 환경 구성

추천 커밋명:

```text
build: FastAPI 추론 서버 Docker 실행 환경 구성
```

주요 작업:

- 우선 backend만 대상으로 `Dockerfile`과 `.dockerignore`를 만든다.
- 학습 데이터, W&B 로그, notebook, 로컬 가상환경을 image에서 제외한다.
- checkpoint는 Git에 없으므로 기본적으로 read-only volume mount하고
  `MODEL_PATH`로 위치를 전달한다.
- container는 CPU에서도 실행 가능해야 하며 `map_location`을 사용한다.
- Uvicorn host를 `0.0.0.0`, port를 8000으로 설정한다.
- 필요할 때만 후속 커밋으로 web/api `docker-compose.yml`을 추가한다.

검증 기준:

- 새 환경에서 image build가 성공한다.
- mounted checkpoint로 `/health`와 `/predict`가 정상 동작한다.
- checkpoint가 없을 때 container log에 명확한 원인이 보인다.
- Docker CPU 추론 결과의 Top-3가 로컬 결과와 허용 오차 내에서 같다.

### 커밋 12. 문서와 최종 End-to-End 검증 정리

추천 커밋명:

```text
docs: 서비스 실행 방법과 검증 절차 정리
```

주요 작업:

- README에 현재 10클래스 데이터, 최종 모델, 전체 아키텍처를 반영한다.
- Windows/macOS/Linux backend 실행법, frontend 실행법, Swagger 주소를 적는다.
- checkpoint 배치·mount 방법과 `.env.example` 복사 방법을 적는다.
- API request/response 예시와 Softmax 점수 해석 주의를 적는다.
- 자동·수동 검증 체크리스트와 확인된 결과를 기록한다.
- 병변 의료 문구 검수, 데이터 생성 provenance 등 남은 TODO를 숨기지 않는다.

최종 검증 기준:

- backend 자동 테스트가 모두 성공한다.
- frontend lint/build가 성공한다.
- `/health`, JPG/JPEG/PNG `/predict`, `/lesions`를 확인한다.
- 잘못된 파일, 손상 이미지, API down, checkpoint 없음, mapping 불일치를 확인한다.
- 브라우저에서 `업로드 -> preview -> 분석 -> Top-3 -> 병변 정보` 흐름을 확인한다.
- 선택 시 Grad-CAM과 Docker CPU 실행까지 확인한다.

## 4. 권장 테스트 배치

테스트만 마지막에 몰아넣지 않고 각 기능 커밋에 함께 넣는다.

```text
tests/
├── test_class_mapping.py
├── test_inference.py
├── test_api_health.py
├── test_api_predict.py
└── test_lesions.py
```

- 단위/API 테스트에서는 작은 임시 checkpoint 또는 inference mock을 사용하여
  매번 16MB 실제 모델을 로드하지 않는다.
- 실제 checkpoint smoke test는 별도 표시하여 로컬이나 CI에서 선택 실행한다.
- frontend는 최소 lint/build를 필수로 하고, 시간이 허용되면 upload/result 흐름의
  component test를 추가한다.

## 5. 우선순위와 완료 지점

### 1차 필수 완성

커밋 1~8까지 완료하면 핵심 데모인 아래 흐름이 완성된다.

```text
Image -> Next.js -> FastAPI -> EfficientNet-B0 -> Top-3 + Lesion Info -> UI
```

### 2차 정보 탐색

커밋 9에서 `/lesions`를 완성한다.

### 3차 선택 기능과 배포

커밋 10의 Grad-CAM과 커밋 11의 Docker는 필수 흐름이 안정화된 후 진행한다.
일정이 부족하면 Grad-CAM을 먼저 제외하고 backend Docker를 우선하는 편이 실제
시연 재현성에 유리하다.

## 6. 커밋 체크 규칙

- 한 커밋에서 backend와 frontend의 서로 무관한 작업을 섞지 않는다.
- 각 커밋 직전 `git diff`로 사용자의 기존 변경을 침범하지 않았는지 확인한다.
- 기능 커밋은 관련 테스트 또는 실행 검증을 반드시 포함한다.
- checkpoint, `.env`, 데이터 파일은 커밋하지 않는다.
- API schema를 바꾸는 커밋에서는 backend schema와 frontend type을 같은 커밋에서
  함께 바꾸거나, 하위 호환 상태를 유지한다.
- Grad-CAM, Docker Compose처럼 선택 기능은 필수 흐름과 별도 커밋으로 유지한다.
