export interface PredictionSummary {
  class: string;
  name_ko: string;
  name_en: string;
  confidence: number;
}

export interface TopPrediction {
  rank: number;
  class: string;
  name_ko: string;
  probability: number;
}

export interface LesionInformation {
  class: string;
  name_ko: string;
  name_en: string;
  category: string;
  description: string;
  features: string[];
  precautions: string[];
}

export interface PredictResponse {
  prediction: PredictionSummary;
  top3: TopPrediction[];
  inference_time_ms: number;
  information: LesionInformation;
  gradcam: { image_base64: string } | null;
  disclaimer: string;
}

export interface LesionsResponse {
  lesions: LesionInformation[];
  disclaimer: string;
}
