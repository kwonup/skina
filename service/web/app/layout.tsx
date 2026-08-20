import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";

import "./globals.css";


export const metadata: Metadata = {
  title: "skina | 피부 이미지 AI 분석",
  description: "피부 이미지를 10개 유형으로 분류하는 학습 프로젝트 서비스",
};


function Header() {
  return (
    <header className="site-header">
      <div className="header-inner">
        <Link className="brand" href="/" aria-label="skina 홈">
          <span className="brand-mark" aria-hidden="true">s</span>
          <span>skina</span>
        </Link>
        <nav className="main-nav" aria-label="주요 메뉴">
          <Link href="/">이미지 분석</Link>
          <Link href="/lesions">병변 정보</Link>
        </nav>
      </div>
    </header>
  );
}


export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="ko">
      <body>
        <Header />
        {children}
        <footer className="site-footer">
          <div className="footer-inner">
            <strong>skina</strong>
            <p>학습 프로젝트용 피부 이미지 분류 서비스</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
