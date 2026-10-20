#!/usr/bin/env python3
"""
실제 문서(매뉴얼, PDF, 텍스트 파일)에서 전문적인 정보를 추출하는 도구

사용 예시:
    # 텍스트 파일에서 추출
    python extract_from_docs.py text --file manual.txt --keyword ALD

    # PDF에서 추출 (PyPDF2 필요)
    python extract_from_docs.py pdf --file manual.pdf --keyword MFC

    # 여러 키워드 한번에
    python extract_from_docs.py text --file manual.txt --keywords ALD,Precursor,Purge
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Set

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ~/ald-rag-lab
DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"

# PDF 처리를 위한 선택적 import
try:
    import PyPDF2  # type: ignore
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def load_docs() -> Dict[str, List[Dict[str, str]]]:
    """docs_ald.json 로드"""
    if not DOCS_PATH.exists():
        return {}
    
    import json
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("documents", {})


def save_docs(docs: Dict[str, List[Dict[str, str]]]) -> None:
    """docs_ald.json 저장"""
    import json
    DOCS_PATH.parent.mkdir(exist_ok=True, parents=True)
    data = {"documents": docs}
    with DOCS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    total = sum(len(texts) for texts in docs.values())
    print(f"[+] 저장 완료: {len(docs)}개 키워드, {total}개 문서")


def extract_from_text(text: str, keywords: List[str]) -> Dict[str, List[str]]:
    """텍스트에서 키워드 관련 문장 추출"""
    results: Dict[str, List[str]] = {kw: [] for kw in keywords}
    
    # 문장 분리 (간단한 방법)
    sentences = re.split(r'[.!?]\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # 너무 짧은 문장 제외
            continue
        
        # 각 키워드에 대해 매칭 확인
        for keyword in keywords:
            # 키워드가 문장에 포함되어 있고, 구체적인 정보가 있는지 확인
            if keyword.lower() in sentence.lower():
                # 구체적인 정보가 있는지 확인 (수치, 범위 등)
                has_specific_info = (
                    re.search(r'\d+', sentence) or  # 숫자 포함
                    re.search(r'[℃°C]|sccm|Torr|W|Hz|%', sentence) or  # 단위 포함
                    len(sentence) > 30  # 충분한 길이
                )
                
                if has_specific_info:
                    # 일반적인 표현 제거
                    if not any(word in sentence for word in ['중요하다', '필요하다', '좋다', '나쁘다', '일반적으로', '보통']):
                        if sentence not in results[keyword]:
                            results[keyword].append(sentence)
    
    return results


def extract_from_pdf(pdf_path: Path, keywords: List[str]) -> Dict[str, List[str]]:
    """PDF에서 키워드 관련 문장 추출"""
    if not PDF_AVAILABLE:
        raise RuntimeError("PyPDF2가 설치되지 않았습니다. 'pip install PyPDF2' 실행하세요.")
    
    results: Dict[str, List[str]] = {kw: [] for kw in keywords}
    
    with pdf_path.open('rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            page_results = extract_from_text(text, keywords)
            
            for keyword in keywords:
                results[keyword].extend(page_results[keyword])
    
    return results


def filter_quality_sentences(sentences: List[str]) -> List[str]:
    """문장 품질 필터링"""
    filtered = []
    
    for sentence in sentences:
        # 너무 짧거나 길면 제외
        if len(sentence) < 20 or len(sentence) > 200:
            continue
        
        # 일반적인 표현 제외
        general_words = ['중요하다', '필요하다', '좋다', '나쁘다', '일반적으로', '보통', '대부분']
        if any(word in sentence for word in general_words):
            continue
        
        # 구체적인 정보가 있는지 확인
        has_specific = (
            re.search(r'\d+', sentence) or
            re.search(r'[℃°C]|sccm|Torr|W|Hz|%|bar|Pa', sentence)
        )
        
        if has_specific:
            filtered.append(sentence)
    
    return filtered


def run_text_mode(file_path: str, keywords: List[str], auto_add: bool = False):
    """텍스트 파일에서 추출"""
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
        return
    
    print(f"[+] 텍스트 파일 읽는 중: {file_path}")
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"[+] 키워드 관련 문장 추출 중: {', '.join(keywords)}")
    results = extract_from_text(text, keywords)
    
    # 품질 필터링
    for keyword in keywords:
        results[keyword] = filter_quality_sentences(results[keyword])
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("[추출된 문장들]")
    print("=" * 80)
    
    docs = load_docs()
    new_count = 0
    
    for keyword in keywords:
        sentences = results[keyword]
        if not sentences:
            print(f"\n📌 {keyword}: (추출된 문장 없음)")
            continue
        
        print(f"\n📌 {keyword}: {len(sentences)}개 문장")
        print("-" * 80)
        
        for i, sentence in enumerate(sentences[:10], 1):  # 최대 10개만 표시
            print(f"  {i}. {sentence}")
        
        if len(sentences) > 10:
            print(f"  ... 외 {len(sentences) - 10}개")
        
        # 자동 추가 또는 확인
        if auto_add:
            if keyword not in docs:
                docs[keyword] = []
            
            for sentence in sentences:
                if {"text": sentence} not in docs[keyword]:
                    docs[keyword].append({"text": sentence})
                    new_count += 1
        else:
            confirm = input(f"\n위 {len(sentences)}개 문장을 '{keyword}' 키워드에 추가하시겠습니까? (y/n): ").strip().lower()
            if confirm == 'y':
                if keyword not in docs:
                    docs[keyword] = []
                
                for sentence in sentences:
                    if {"text": sentence} not in docs[keyword]:
                        docs[keyword].append({"text": sentence})
                        new_count += 1
    
    if new_count > 0:
        save_docs(docs)
        print(f"\n[✓] 총 {new_count}개 문장이 추가되었습니다.")
    else:
        print("\n[!] 추가된 문장이 없습니다.")


def run_pdf_mode(file_path: str, keywords: List[str], auto_add: bool = False):
    """PDF 파일에서 추출"""
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
        return
    
    if not PDF_AVAILABLE:
        print("[ERROR] PyPDF2가 설치되지 않았습니다.")
        print("        설치: pip install PyPDF2")
        return
    
    print(f"[+] PDF 파일 읽는 중: {file_path}")
    results = extract_from_pdf(path, keywords)
    
    # 품질 필터링
    for keyword in keywords:
        results[keyword] = filter_quality_sentences(results[keyword])
    
    # 결과 출력 및 저장 (텍스트 모드와 동일)
    print("\n" + "=" * 80)
    print("[추출된 문장들]")
    print("=" * 80)
    
    docs = load_docs()
    new_count = 0
    
    for keyword in keywords:
        sentences = results[keyword]
        if not sentences:
            print(f"\n📌 {keyword}: (추출된 문장 없음)")
            continue
        
        print(f"\n📌 {keyword}: {len(sentences)}개 문장")
        print("-" * 80)
        
        for i, sentence in enumerate(sentences[:10], 1):
            print(f"  {i}. {sentence}")
        
        if len(sentences) > 10:
            print(f"  ... 외 {len(sentences) - 10}개")
        
        if auto_add:
            if keyword not in docs:
                docs[keyword] = []
            
            for sentence in sentences:
                if {"text": sentence} not in docs[keyword]:
                    docs[keyword].append({"text": sentence})
                    new_count += 1
        else:
            confirm = input(f"\n위 {len(sentences)}개 문장을 '{keyword}' 키워드에 추가하시겠습니까? (y/n): ").strip().lower()
            if confirm == 'y':
                if keyword not in docs:
                    docs[keyword] = []
                
                for sentence in sentences:
                    if {"text": sentence} not in docs[keyword]:
                        docs[keyword].append({"text": sentence})
                        new_count += 1
    
    if new_count > 0:
        save_docs(docs)
        print(f"\n[✓] 총 {new_count}개 문장이 추가되었습니다.")


def parse_args():
    parser = argparse.ArgumentParser(description="문서에서 전문적인 정보 추출")
    sub = parser.add_subparsers(dest="mode", required=True)
    
    # 텍스트 모드
    p_text = sub.add_parser("text", help="텍스트 파일에서 추출")
    p_text.add_argument("--file", type=str, required=True, help="텍스트 파일 경로")
    p_text.add_argument("--keywords", type=str, required=True, help="키워드 (쉼표로 구분)")
    p_text.add_argument("--auto", action="store_true", help="자동 추가 (확인 없이)")
    
    # PDF 모드
    p_pdf = sub.add_parser("pdf", help="PDF 파일에서 추출")
    p_pdf.add_argument("--file", type=str, required=True, help="PDF 파일 경로")
    p_pdf.add_argument("--keywords", type=str, required=True, help="키워드 (쉼표로 구분)")
    p_pdf.add_argument("--auto", action="store_true", help="자동 추가 (확인 없이)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    keywords = [kw.strip() for kw in args.keywords.split(",")]
    
    if args.mode == "text":
        run_text_mode(args.file, keywords, args.auto)
    elif args.mode == "pdf":
        run_pdf_mode(args.file, keywords, args.auto)

