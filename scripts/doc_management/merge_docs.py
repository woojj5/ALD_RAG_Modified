#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
docs_ald.json 문서 관리 통합 스크립트

기능:
1. 새 문서 추가 (NEW_DOCS 리스트에 문서가 있는 경우)
2. 중복 제거 (기존 문서 내 + 새 문서 vs 기존 문서)
3. 키워드 개수별, 종류별 정렬
4. ID 재할당

사용법:
- 새 문서 추가: NEW_DOCS 리스트에 문서 추가 후 실행
- 정렬만: NEW_DOCS를 빈 리스트([])로 두고 실행
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 파일 경로
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ~/ald-rag-lab
JSON_FILE = BASE_DIR / "docs" / "docs_ald.json"

# 추가할 새 문서들 (빈 리스트면 정렬만 수행)
# 새 문서를 추가하려면 아래 리스트에 문서를 추가하세요
# 예시:
# NEW_DOCS = [
#     {
#         "id": 1,  # id는 무시됨 (자동 할당)
#         "keywords": ["ALD"],
#         "text": "새로운 문서 내용"
#     },
# ]
NEW_DOCS = []


def normalize_text(text):
    """텍스트를 정규화하여 비교 (공백 제거, 소문자 변환 등)"""
    # 공백 제거 및 소문자 변환
    return text.strip().lower().replace(" ", "").replace("\n", "").replace("\t", "")


def is_duplicate_text(text1: str, text2: str) -> bool:
    """두 텍스트가 중복인지 확인"""
    normalized1 = normalize_text(text1)
    normalized2 = normalize_text(text2)
    
    # 완전 일치
    if normalized1 == normalized2:
        return True
    
    # 유사도가 높은 경우 (85% 이상 일치)
    if len(normalized1) > 0 and len(normalized2) > 0:
        common_chars = sum(1 for c in normalized1 if c in normalized2)
        similarity = common_chars / max(len(normalized1), len(normalized2))
        if similarity > 0.85:
            return True
    
    return False


def is_duplicate(new_text: str, existing_docs: List[Dict[str, Any]]) -> Tuple[bool, int]:
    """새 텍스트가 기존 문서와 중복되는지 확인"""
    for doc in existing_docs:
        if is_duplicate_text(new_text, doc.get("text", "")):
            return True, doc["id"]
    return False, None


def remove_duplicates(documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """기존 문서 내에서 중복 제거"""
    unique_docs = []
    seen_texts = set()
    removed_count = 0
    
    for doc in documents:
        normalized_text = normalize_text(doc.get("text", ""))
        
        # 이미 본 텍스트인지 확인
        is_dup = False
        for seen_text in seen_texts:
            if is_duplicate_text(normalized_text, seen_text):
                is_dup = True
                break
        
        if not is_dup:
            unique_docs.append(doc)
            seen_texts.add(normalized_text)
        else:
            removed_count += 1
    
    return unique_docs, removed_count


def normalize_keyword(keyword, existing_docs):
    """키워드를 기존 형식에 맞게 정규화"""
    kw_lower = keyword.lower()
    
    # 기존 문서에서 같은 키워드 찾기 (대소문자 무시)
    for doc in existing_docs:
        for ekw in doc.get("keywords", []):
            if ekw.lower() == kw_lower:
                # 기존 키워드 형식 반환
                return ekw
    
    # 없으면 원본 반환
    return keyword


def merge_documents():
    """문서 병합 및 정렬 실행"""
    # 기존 파일 읽기
    if not JSON_FILE.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {JSON_FILE}")
        return
    
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 원본 데이터 백업용 복사
    original_data = json.loads(json.dumps(data))
    
    existing_docs = data.get("documents", [])
    
    print(f"📄 기존 문서 수: {len(existing_docs)}")
    
    # 중복 제거 (기존 문서 내)
    print("🔍 기존 문서 내 중복 검사 중...")
    unique_docs, removed_count = remove_duplicates(existing_docs)
    
    if removed_count > 0:
        print(f"🗑️  중복 문서 {removed_count}개 제거됨")
        print(f"📄 중복 제거 후 문서 수: {len(unique_docs)}개")
    else:
        print("✅ 기존 문서 내 중복 없음")
    
    existing_docs = unique_docs
    
    # 새 문서 추가 여부 확인
    if NEW_DOCS:
        print(f"➕ 추가할 새 문서 수: {len(NEW_DOCS)}")
    else:
        print("ℹ️  추가할 새 문서 없음 (정렬만 수행)")
    
    print("-" * 60)
    
    # 키워드 개수별로 그룹화하여 삽입 위치 파악
    keyword_count_groups = {}
    for doc in existing_docs:
        count = len(doc.get("keywords", []))
        if count not in keyword_count_groups:
            keyword_count_groups[count] = []
        keyword_count_groups[count].append(doc)
    
    added_count = 0
    skipped_count = 0
    skipped_docs = []
    docs_to_add = []  # 추가할 문서들을 모아서 나중에 정리
    
    # 새 문서 추가 (NEW_DOCS가 있는 경우만)
    if NEW_DOCS:
        for new_doc in NEW_DOCS:
            is_dup, existing_id = is_duplicate(new_doc["text"], existing_docs)
            
            if is_dup:
                print(f"⏭️  건너뜀 (중복): ID {existing_id}와 유사")
                print(f"   텍스트: {new_doc['text'][:60]}...")
                skipped_count += 1
                skipped_docs.append(new_doc)
            else:
                # 키워드 정규화 (기존 형식에 맞춤)
                normalized_keywords = []
                for kw in new_doc["keywords"]:
                    normalized_kw = normalize_keyword(kw, existing_docs)
                    normalized_keywords.append(normalized_kw)
                
                new_doc["keywords"] = normalized_keywords
                docs_to_add.append(new_doc)
                added_count += 1
    
    # 새 문서를 키워드 개수별 그룹에 추가 (NEW_DOCS가 있는 경우만)
    if docs_to_add:
        new_docs_by_count = {}
        for doc in docs_to_add:
            count = len(doc.get("keywords", []))
            if count not in new_docs_by_count:
                new_docs_by_count[count] = []
            new_docs_by_count[count].append(doc)
        
        # 각 키워드 개수 그룹에 새 문서 추가
        for count in sorted(new_docs_by_count.keys()):
            if count not in keyword_count_groups:
                keyword_count_groups[count] = []
            
            # 같은 키워드 개수 내에서 정렬 (키워드 종류별)
            def sort_key(doc):
                keywords = sorted(doc.get("keywords", []))
                return tuple(keywords) + (doc.get("text", ""),)
            
            new_docs_by_count[count].sort(key=sort_key)
            
            keyword_count_groups[count].extend(new_docs_by_count[count])
            print(f"✅ {count}개 키워드 문서 {len(new_docs_by_count[count])}개 추가 예정")
    
    # 키워드 개수별로 재정렬하고 ID 재할당
    reorganized = []
    new_id = 1
    
    for keyword_count in sorted(keyword_count_groups.keys()):
        docs_in_group = keyword_count_groups[keyword_count]
        
        # 같은 키워드 개수 내에서 정렬 (키워드 종류별)
        def sort_key(doc):
            keywords = sorted(doc.get("keywords", []))
            return tuple(keywords) + (doc.get("text", ""),)
        
        docs_in_group.sort(key=sort_key)
        
        for doc in docs_in_group:
            doc["id"] = new_id
            reorganized.append(doc)
            new_id += 1
    
    # 추가된 문서 정보 출력 (NEW_DOCS가 있는 경우만)
    if docs_to_add:
        print("-" * 60)
        for doc in docs_to_add:
            # 재정렬 후 ID 찾기
            for reorg_doc in reorganized:
                if (reorg_doc.get("text") == doc.get("text") and 
                    reorg_doc.get("keywords") == doc.get("keywords")):
                    print(f"✅ 추가됨: ID {reorg_doc['id']}, 키워드: {reorg_doc['keywords']}")
                    print(f"   텍스트: {reorg_doc['text'][:60]}...")
                    break
    
    # 백업 파일 관리 (저장 전에 백업)
    BACKUP_FILE = JSON_FILE.with_suffix(".json.backup")
    if BACKUP_FILE.exists():
        backup2 = JSON_FILE.parent / "docs_ald.json.backup2"
        if backup2.exists():
            backup2.unlink()
        BACKUP_FILE.rename(backup2)
        print(f"💾 기존 백업을 backup2로 이동")
    
    # 현재 파일을 백업으로 저장 (원본 데이터)
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)
    print(f"💾 현재 파일을 백업으로 저장: {BACKUP_FILE}")
    
    # 결과 저장
    data["documents"] = reorganized
    
    # 새 파일 저장
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("-" * 60)
    print(f"✅ 완료!")
    if removed_count > 0:
        print(f"   기존 문서 중복 제거: {removed_count}개")
    if NEW_DOCS:
        print(f"   추가된 문서: {added_count}개")
        print(f"   건너뛴 문서: {skipped_count}개")
    print(f"   총 문서 수: {len(reorganized)}개")
    
    # 키워드 개수별 통계 출력
    print(f"\n📋 키워드 개수별 분류:")
    for count in sorted(keyword_count_groups.keys()):
        start_id = sum(len(keyword_count_groups[c]) for c in sorted(keyword_count_groups.keys()) if c < count) + 1
        end_id = start_id + len(keyword_count_groups[count]) - 1
        print(f"   {count}개 키워드: ID {start_id} ~ {end_id} ({len(keyword_count_groups[count])}개)")
    
    if skipped_docs:
        print(f"\n⚠️  건너뛴 문서 목록:")
        for doc in skipped_docs:
            print(f"   - {doc['text'][:60]}...")


if __name__ == "__main__":
    merge_documents()
