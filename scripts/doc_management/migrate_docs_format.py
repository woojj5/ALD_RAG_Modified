#!/usr/bin/env python3
"""
docs_ald.json 파일을 키워드별 그룹화 구조로 변환하는 스크립트

기존 구조: {"documents": [{"keyword": "ALD", "text": "..."}, ...]}
새 구조: {"documents": {"ALD": [{"text": "..."}, ...], ...}}
"""

import json
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ~/ald-rag-lab
DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"
BACKUP_PATH = BASE_DIR / "docs" / "docs_ald.json.backup"


def migrate():
    """기존 구조를 새 구조로 변환"""
    if not DOCS_PATH.exists():
        print(f"[ERROR] {DOCS_PATH} 파일이 없습니다.")
        return False

    # 백업 생성
    print(f"[+] 백업 생성: {BACKUP_PATH}")
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        backup_data = f.read()
    BACKUP_PATH.write_text(backup_data, encoding="utf-8")

    # 기존 데이터 로드
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        old_data = json.load(f)

    # 기존 구조 파싱
    if isinstance(old_data, dict) and "documents" in old_data:
        items = old_data["documents"]
    elif isinstance(old_data, list):
        items = old_data
    else:
        print("[ERROR] 알 수 없는 파일 구조입니다.")
        return False

    # 새 구조로 변환
    new_docs: Dict[str, List[Dict[str, str]]] = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        keyword = str(item.get("keyword", "unknown")).strip() or "unknown"
        text = str(item.get("text", "")).strip()

        if not text:
            continue

        if keyword not in new_docs:
            new_docs[keyword] = []

        new_docs[keyword].append({"text": text})

    # 새 구조로 저장
    new_data = {"documents": new_docs}

    with DOCS_PATH.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    total_docs = sum(len(texts) for texts in new_docs.values())
    print(f"[+] 변환 완료!")
    print(f"    - 키워드 수: {len(new_docs)}")
    print(f"    - 총 문서 수: {total_docs}")
    print(f"    - 백업 파일: {BACKUP_PATH}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("docs_ald.json 구조 변환 스크립트")
    print("=" * 60)
    print()
    
    if migrate():
        print("\n[✓] 변환이 성공적으로 완료되었습니다!")
    else:
        print("\n[✗] 변환에 실패했습니다.")

