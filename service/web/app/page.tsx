import ImageAnalyzer from "@/components/ImageAnalyzer";


export default function Home() {
  return (
    <main>
      <section className="hero section-shell">
        <div className="eyebrow">SKIN IMAGE CLASSIFICATION</div>
        <h1>피부 이미지를 올리고<br />AI 예측 결과를 확인하세요</h1>
        <p className="hero-copy">
          학습된 이미지 분류 모델이 10개 피부 병변 유형 중 가장 높은 점수를 받은
          결과와 Top-3를 보여드립니다.
        </p>
        <div className="service-chips" aria-label="서비스 특징">
          <span>10개 유형 분류</span>
          <span>Top-3 결과</span>
          <span>병변 정보 제공</span>
        </div>
      </section>

      <section className="section-shell analysis-section" aria-labelledby="analysis-title">
        <div className="section-heading">
          <span className="step-number">01</span>
          <div>
            <h2 id="analysis-title">이미지 분석</h2>
            <p>JPG, JPEG, PNG 파일을 선택해 주세요.</p>
          </div>
        </div>
        <ImageAnalyzer />
      </section>

      <aside className="section-shell disclaimer-card" aria-label="이용 안내">
        <span className="notice-mark" aria-hidden="true">i</span>
        <div>
          <strong>이용 전 확인해 주세요</strong>
          <p>
            본 결과는 학습 프로젝트의 이미지 분류 모델이 제공하는 참고용 예측
            결과이며, 의료적 진단을 대체하지 않습니다.
          </p>
        </div>
      </aside>
    </main>
  );
}
