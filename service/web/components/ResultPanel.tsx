import type { LesionInformation, PredictResponse } from "@/lib/types";


function displayName(className: string, nameKo?: string): string {
  return nameKo?.trim() || className.replaceAll("_", " ");
}


function InformationSection({ information }: { information: LesionInformation }) {
  const hasVerifiedInformation = Boolean(
    information.description ||
    information.category ||
    information.features.length ||
    information.precautions.length,
  );

  if (!hasVerifiedInformation) {
    return (
      <div className="information-empty">
        <strong>검수된 병변 정보 준비 중</strong>
        <p>의료 정보는 검증된 자료가 확보된 후 제공됩니다.</p>
      </div>
    );
  }

  return (
    <div className="information-grid">
      {information.description && (
        <section className="information-block information-description">
          <span className="info-label">병변 설명</span>
          <p>{information.description}</p>
        </section>
      )}
      {information.features.length > 0 && (
        <section className="information-block">
          <span className="info-label">주요 특징</span>
          <ul>{information.features.map((item) => <li key={item}>{item}</li>)}</ul>
        </section>
      )}
      {information.precautions.length > 0 && (
        <section className="information-block caution-block">
          <span className="info-label">주의사항</span>
          <ul>{information.precautions.map((item) => <li key={item}>{item}</li>)}</ul>
        </section>
      )}
    </div>
  );
}


export default function ResultPanel({ result }: { result: PredictResponse }) {
  const topPrediction = result.prediction;
  const score = (topPrediction.confidence * 100).toFixed(1);

  return (
    <section className="result-section" aria-labelledby="result-title">
      <div className="section-heading result-heading">
        <span className="step-number">02</span>
        <div>
          <h2 id="result-title">분석 결과</h2>
          <p>모델이 이미지에 부여한 분류 점수입니다.</p>
        </div>
      </div>

      <div className="result-card">
        <div className="top-result">
          <span className="result-kicker">가장 높게 예측된 유형</span>
          <h3>{displayName(topPrediction.class, topPrediction.name_ko)}</h3>
          {topPrediction.name_en && <p className="english-name">{topPrediction.name_en}</p>}
          <div className="score-line">
            <strong>{score}%</strong>
            <span>모델 예측 점수</span>
          </div>
          <p className="prediction-language">
            이 유형으로 가장 높게 예측되었습니다. 확정 진단을 의미하지 않습니다.
          </p>
        </div>

        <div className="top-three" aria-labelledby="top-three-title">
          <div className="subsection-title">
            <h3 id="top-three-title">Top-3</h3>
            <span>{result.inference_time_ms.toFixed(1)} ms</span>
          </div>
          <ol>
            {result.top3.map((item) => {
              const percentage = Math.max(0, Math.min(100, item.probability * 100));
              return (
                <li key={item.class}>
                  <div className="prediction-row">
                    <span className="rank">{item.rank}</span>
                    <span className="prediction-name">
                      {displayName(item.class, item.name_ko)}
                    </span>
                    <strong>{percentage.toFixed(1)}%</strong>
                  </div>
                  <div className="score-track" aria-hidden="true">
                    <span style={{ width: `${percentage}%` }} />
                  </div>
                </li>
              );
            })}
          </ol>
        </div>
      </div>

      <div className="result-information-card">
        <div className="subsection-title">
          <h3>병변 정보</h3>
          {result.information.category && <span>{result.information.category}</span>}
        </div>
        <InformationSection information={result.information} />
      </div>

      {result.gradcam?.image_base64 && (
        <div className="gradcam-card">
          <div>
            <span className="info-label">보조 시각화</span>
            <h3>Grad-CAM</h3>
            <p>모델이 분류에 참고한 영역을 색상으로 표현한 보조 자료입니다.</p>
          </div>
          {/* API가 생성한 일회성 data URL이므로 Next 이미지 최적화를 사용하지 않는다. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${result.gradcam.image_base64}`}
            alt="모델이 참고한 이미지 영역을 표시한 Grad-CAM"
          />
        </div>
      )}

      <div className="disclaimer-card result-disclaimer">
        <span className="notice-mark" aria-hidden="true">i</span>
        <div>
          <strong>참고용 예측 결과입니다</strong>
          <p>{result.disclaimer}</p>
        </div>
      </div>
    </section>
  );
}
