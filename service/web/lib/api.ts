import type { LesionsResponse, PredictResponse } from "@/lib/types";


const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");


export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}


async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return response.json() as Promise<T>;
  }

  let message = "요청을 처리하지 못했습니다.";
  try {
    const body = (await response.json()) as { detail?: string };
    if (body.detail) message = body.detail;
  } catch {
    // JSON이 아닌 오류 응답에는 기본 메시지를 사용한다.
  }
  throw new ApiError(message, response.status);
}


export async function predictImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("image", file);
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });
    return parseResponse<PredictResponse>(response);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("분석 서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.");
  }
}


export async function getLesions(): Promise<LesionsResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/lesions`, {
      cache: "no-store",
    });
    return parseResponse<LesionsResponse>(response);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("병변 정보를 불러오지 못했습니다.");
  }
}
