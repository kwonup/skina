"use client";

import Image from "next/image";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

import { predictImage } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";
import ResultPanel from "@/components/ResultPanel";


const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png"]);
const ALLOWED_EXTENSIONS = new Set(["jpg", "jpeg", "png"]);


function validateFile(file: File): string | null {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ALLOWED_TYPES.has(file.type) || !ALLOWED_EXTENSIONS.has(extension)) {
    return "JPG, JPEG, PNG 이미지 파일만 선택할 수 있습니다.";
  }
  if (file.size === 0) return "비어 있는 파일은 분석할 수 없습니다.";
  if (file.size > MAX_FILE_SIZE) return "이미지는 최대 10MB까지 업로드할 수 있습니다.";
  return null;
}


export default function ImageAnalyzer() {
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  function replacePreview(nextUrl: string | null) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = nextUrl;
    setPreviewUrl(nextUrl);
  }

  function selectFile(nextFile: File | undefined) {
    if (!nextFile) return;
    const validationError = validateFile(nextFile);
    if (validationError) {
      setError(validationError);
      setFile(null);
      replacePreview(null);
      setResult(null);
      if (inputRef.current) inputRef.current.value = "";
      return;
    }
    setFile(nextFile);
    replacePreview(URL.createObjectURL(nextFile));
    setError(null);
    setResult(null);
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    if (!isLoading) selectFile(event.dataTransfer.files?.[0]);
  }

  async function handleAnalyze() {
    if (!file || isLoading) return;
    setIsLoading(true);
    setError(null);
    try {
      setResult(await predictImage(file));
    } catch (caughtError) {
      setResult(null);
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "이미지를 분석하지 못했습니다. 다시 시도해 주세요.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="analyzer">
      <div
        className={`upload-card interactive ${isDragging ? "is-dragging" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          if (!isLoading) setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node)) {
            setIsDragging(false);
          }
        }}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept=".jpg,.jpeg,.png,image/jpeg,image/png"
          onChange={handleInputChange}
          disabled={isLoading}
          aria-label="분석할 피부 이미지 선택"
        />

        {file && previewUrl ? (
          <div className="selected-file">
            <div className="image-preview">
              <Image
                src={previewUrl}
                alt="선택한 피부 이미지 미리보기"
                fill
                sizes="(max-width: 640px) 80vw, 420px"
                unoptimized
              />
            </div>
            <div className="file-meta">
              <strong>{file.name}</strong>
              <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            <div className="button-row">
              <button
                type="button"
                className="button button-secondary"
                onClick={() => inputRef.current?.click()}
                disabled={isLoading}
              >
                다른 이미지 선택
              </button>
              <button
                type="button"
                className="button button-primary analyze-button"
                onClick={handleAnalyze}
                disabled={isLoading}
              >
                {isLoading ? (
                  <><span className="spinner" aria-hidden="true" />분석 중...</>
                ) : "분석하기"}
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="upload-symbol" aria-hidden="true">+</div>
            <strong>{isDragging ? "여기에 이미지를 놓아주세요" : "분석할 피부 이미지를 선택하세요"}</strong>
            <p>파일을 끌어다 놓거나 버튼을 눌러 주세요. 최대 10MB</p>
            <button
              type="button"
              className="button button-primary"
              onClick={() => inputRef.current?.click()}
              disabled={isLoading}
            >
              이미지 선택
            </button>
          </>
        )}
      </div>

      {error && <div className="error-message" role="alert">{error}</div>}
      {result && <ResultPanel result={result} />}
    </div>
  );
}
