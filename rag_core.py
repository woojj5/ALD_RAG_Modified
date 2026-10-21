from __future__ import annotations

import json
import os
import hashlib
import re
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING
from collections import OrderedDict

import numpy as np  # type: ignore

if TYPE_CHECKING:
    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

# ==============================
# 0) 경로 / 상수 설정
# ==============================

BASE_DIR = Path(__file__).resolve().parent           # ~/ald-rag-lab
DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"
SYNONYMS_PATH = BASE_DIR / "synonyms.json"
FEEDBACK_PATH = BASE_DIR / "feedback" / "feedback_data.json"

EMBED_MODEL_NAME = "thenlper/gte-small"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_BASE_URL = os.getenv("ALD_RAG_LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("ALD_RAG_LLM_API_KEY", "")
LLM_REMOTE_MODEL = os.getenv("ALD_RAG_LLM_MODEL", LLM_MODEL_NAME)
LLM_REQUEST_TIMEOUT = float(os.getenv("ALD_RAG_LLM_TIMEOUT", "60"))
# Fine-tuned 모델 경로 (None이면 기본 모델 사용)
FINETUNED_MODEL_PATH = BASE_DIR / "finetuning" / "models" / "qwen-ald-lora"

# 전역 상태
DOCS: List[str] = []
DOC_KEYWORDS: List[str] = []
DOC_ITEMS: List[Dict[str, Any]] = []
DOC_EMBEDS: Optional[np.ndarray] = None

# 해시 테이블 인덱스 (키워드 -> 문서 인덱스 리스트)
KEYWORD_INDEX: Dict[str, List[int]] = {}  # keyword -> [doc_idx1, doc_idx2, ...]
KEYWORD_INDEX_NORMALIZED: Dict[str, List[int]] = {}  # normalized keyword -> [doc_idx1, doc_idx2, ...]

DEVICE: Optional[torch.device] = None
SEMB_MODEL: Optional[SentenceTransformer] = None
TOKENIZER: Optional[AutoTokenizer] = None
LLM: Optional[AutoModelForCausalLM] = None

MODEL_INFO: Dict[str, Any] = {}

# 동의어 매핑
SYNONYM_GROUPS: List[List[str]] = []
SYNONYM_MAP: Dict[str, str] = {}  # keyword -> normalized keyword

# 피드백 기반 점수 조정 (문서 인덱스 -> 점수 보정값)
FEEDBACK_SCORES: Dict[int, float] = {}  # 문서 인덱스 -> 점수 보정값 (양수: 보너스, 음수: 페널티)
FEEDBACK_LAST_MODIFIED: float = 0.0  # 피드백 파일 마지막 수정 시간 (캐싱용)

# 쿼리 결과 캐시 (LRU 캐시, 최대 100개)
QUERY_CACHE: OrderedDict[
    str, Tuple[str, List[Tuple[str, float, str, Optional[int]]], float, List[str]]
] = OrderedDict()
MAX_CACHE_SIZE = 100


# ==============================
# 1) JSON 문서 로딩
# ==============================

def load_docs(path: Path = DOCS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"docs 파일이 없음: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    docs, keywords, pairs = [], [], []

    # 새 구조: {"documents": [{"id": 1, "keywords": ["ALD"], "text": "..."}, ...]}
    if isinstance(raw, dict) and "documents" in raw:
        documents = raw["documents"]
        
        if isinstance(documents, list):
            # 새 구조: keywords가 배열인 형태
            for i, item in enumerate(documents):
                if not isinstance(item, dict):
                    print(f"[!] 경고: 문서 #{i} 형식 오류. dict 아님 → 건너뜀")
                    continue

                text = str(item.get("text", "")).strip()
                item_keywords = item.get("keywords", [])
                
                # keywords가 배열인 경우
                if isinstance(item_keywords, list):
                    if not text or not item_keywords:
                        continue
                    
                    # 각 키워드마다 별도의 항목으로 추가 (검색 호환성)
                    for kw in item_keywords:
                        kw = str(kw).strip()
                        if not kw:
                            continue
                        docs.append(text)
                        keywords.append(kw)
                        pairs.append({"text": text, "keyword": kw, "id": item.get("id")})
                
                # 기존 구조 호환: keyword가 단일 문자열인 경우
                elif isinstance(item_keywords, str):
                    kw = item_keywords.strip()
                    if not text or not kw:
                        continue
                    docs.append(text)
                    keywords.append(kw)
                    pairs.append({"text": text, "keyword": kw, "id": item.get("id")})
                
                # 기존 구조 호환: keyword 필드가 있는 경우
                elif "keyword" in item:
                    kw = str(item.get("keyword", "unknown")).strip()
                    if not text or not kw:
                        continue
                    docs.append(text)
                    keywords.append(kw)
                    pairs.append({"text": text, "keyword": kw, "id": item.get("id")})
        
        # 기존 구조 호환: documents가 dict인 경우 (키워드별 그룹화)
        elif isinstance(documents, dict):
            for keyword, text_list in documents.items():
                if not isinstance(text_list, list):
                    continue
                
                for item in text_list:
                    if isinstance(item, dict):
                        text = str(item.get("text", "")).strip()
                    elif isinstance(item, str):
                        text = item.strip()
                    else:
                        continue
                    
                    if not text:
                        continue
                    
                    docs.append(text)
                    keywords.append(keyword)
                    pairs.append({"text": text, "keyword": keyword})
        else:
            raise ValueError("docs_ald.json 형식 오류 — documents는 list 또는 dict여야 함")
    else:
        raise ValueError("docs_ald.json 형식 오류 — 반드시 { 'documents': [...] } 형태여야 함")

    print(f"[+] 문서 로딩 완료 — 총 {len(docs)}개")
    return docs, keywords, pairs


# ==============================
# 2) 모델 초기화
# ==============================

def _build_keyword_index():
    """키워드 기반 해시 테이블 인덱스 구축"""
    global KEYWORD_INDEX, KEYWORD_INDEX_NORMALIZED
    
    KEYWORD_INDEX = {}
    KEYWORD_INDEX_NORMALIZED = {}
    
    for idx, keyword in enumerate(DOC_KEYWORDS):
        # 정확한 키워드 매칭
        if keyword not in KEYWORD_INDEX:
            KEYWORD_INDEX[keyword] = []
        KEYWORD_INDEX[keyword].append(idx)
        
        # 정규화된 키워드 매칭 (동의어 지원)
        normalized = normalize_keyword(keyword)
        if normalized not in KEYWORD_INDEX_NORMALIZED:
            KEYWORD_INDEX_NORMALIZED[normalized] = []
        KEYWORD_INDEX_NORMALIZED[normalized].append(idx)
    
    print(f"[+] 키워드 인덱스 구축 완료: {len(KEYWORD_INDEX)}개 키워드")


_MODEL_INIT_LOCK = threading.Lock()
_MODELS_INITIALIZED = False
_LOCAL_LLM_LOCK = threading.Lock()


def _init_models_if_needed():
    """Initialize models once and block callers until initialization completes."""
    global _MODELS_INITIALIZED

    if _MODELS_INITIALIZED:
        load_feedback_scores()
        return

    with _MODEL_INIT_LOCK:
        if _MODELS_INITIALIZED:
            load_feedback_scores()
            return
        _initialize_models()
        _MODELS_INITIALIZED = True



def _load_local_llm_if_needed():
    """Load the Qwen 7B base model and its LoRA adapter once, on demand."""
    global LLM, TOKENIZER
    if LLM is not None and TOKENIZER is not None:
        return
    with _LOCAL_LLM_LOCK:
        if LLM is not None and TOKENIZER is not None:
            return
        if DEVICE is None:
            raise RuntimeError("RAG device is not initialized")
        if not FINETUNED_MODEL_PATH.is_dir():
            raise FileNotFoundError(f"Fine-tuned adapter not found: {FINETUNED_MODEL_PATH}")
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        print(f"[+] Loading local base model: {LLM_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, torch_dtype=DTYPE,
            device_map="auto" if DEVICE.type != "cpu" else None,
            trust_remote_code=True,
        )
        try:
            from peft import PeftModel  # type: ignore
        except ImportError as error:
            raise RuntimeError("peft is required to load the QLoRA adapter") from error
        print(f"[+] Loading QLoRA adapter: {FINETUNED_MODEL_PATH}")
        model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)
        model.eval()
        TOKENIZER = tokenizer
        LLM = model
        print(f"[+] Local Qwen 7B QLoRA model ready (device={DEVICE})")
def _initialize_models():
    global DOCS, DOC_KEYWORDS, DOC_ITEMS, DOC_EMBEDS
    global SEMB_MODEL, TOKENIZER, LLM, DEVICE, DTYPE, MODEL_INFO
    global KEYWORD_INDEX, KEYWORD_INDEX_NORMALIZED

    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    # 동의어 로딩
    load_synonyms()
    
    # 문서 로딩
    DOCS, DOC_KEYWORDS, DOC_ITEMS = load_docs()
    
    # 키워드 인덱스 구축 (해시 테이블)
    _build_keyword_index()

    # 임베딩 모델
    # 임베딩 모델
    print(f"[+] Embedding model 로딩 중: {EMBED_MODEL_NAME}")
    # 메타 텐서 오류를 방지하기 위해 device를 명시적으로 지정합니다.
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        SEMB_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device=embed_device)
    except Exception:
        # 오류 발생 시 CPU로 강제 로드 후 디바이스 이동
        SEMB_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        if torch.cuda.is_available():
            SEMB_MODEL = SEMB_MODEL.to("cuda")
    
    DOC_EMBEDS = SEMB_MODEL.encode(
        DOCS,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    print("[+] 문서 임베딩 shape:", DOC_EMBEDS.shape)

    # 디바이스 결정
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        DTYPE = torch.float16
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        DTYPE = torch.float16
    else:
        DEVICE = torch.device("cpu")
        DTYPE = torch.float32

    if LLM_BASE_URL:
        TOKENIZER = None
        LLM = None
        print(f"[+] Small-model endpoint configured: {LLM_BASE_URL}")
    else:
        _load_local_llm_if_needed()
    MODEL_INFO.update({
        "num_docs": len(DOCS),
        "keywords": sorted(list(set(DOC_KEYWORDS))),
        "embed_model": EMBED_MODEL_NAME,
        "llm_model": f"{LLM_REMOTE_MODEL} + {LLM_MODEL_NAME} QLoRA" if LLM_BASE_URL else f"{LLM_MODEL_NAME} QLoRA",
        "llm_backend": "complexity_router" if LLM_BASE_URL else "local_transformers",
        "device": str(DEVICE),
    })


def reload_documents():
    """
    docs_ald.json 파일을 다시 로드하고 임베딩을 재생성합니다.
    모델은 이미 로드되어 있어야 합니다 (SEMB_MODEL이 있어야 함).
    """
    global DOCS, DOC_KEYWORDS, DOC_ITEMS, DOC_EMBEDS, MODEL_INFO, SEMB_MODEL
    
    if SEMB_MODEL is None:
        # 모델이 아직 초기화되지 않았으면 전체 초기화
        _init_models_if_needed()
        return
    
    print("[+] 문서 재로딩 중...")
    
    # 문서 다시 로딩
    DOCS, DOC_KEYWORDS, DOC_ITEMS = load_docs()
    
    # 키워드 인덱스 재구축 (해시 테이블)
    _build_keyword_index()
    
    # 임베딩 재생성
    DOC_EMBEDS = SEMB_MODEL.encode(
        DOCS,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")
    QUERY_CACHE.clear()
    
    print(f"[+] 문서 재로딩 완료 — 총 {len(DOCS)}개")
    print(f"[+] 문서 임베딩 shape: {DOC_EMBEDS.shape}")
    
    # MODEL_INFO 업데이트
    MODEL_INFO.update({
        "num_docs": len(DOCS),
        "keywords": sorted(list(set(DOC_KEYWORDS))),
        })
    
    return len(DOCS)


# ==============================
# 3) 동의어 매핑 로딩
# ==============================

def load_synonyms(path: Path = SYNONYMS_PATH):
    """동의어 그룹을 로드하고 매핑을 생성합니다."""
    global SYNONYM_GROUPS, SYNONYM_MAP
    
    if not path.exists():
        print(f"[!] 동의어 파일이 없습니다: {path}. 기본 동의어만 사용합니다.")
        SYNONYM_GROUPS = []
        SYNONYM_MAP = {}
        return
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        SYNONYM_GROUPS = data.get("synonym_groups", [])
        
        # 각 그룹의 첫 번째 키워드를 정규화 키워드로 사용
        SYNONYM_MAP = {}
        for group in SYNONYM_GROUPS:
            if not group:
                continue
            normalized = group[0].lower()
            for keyword in group:
                SYNONYM_MAP[keyword.lower()] = normalized
        
        print(f"[+] 동의어 매핑 로드 완료: {len(SYNONYM_GROUPS)}개 그룹")
    except Exception as e:
        print(f"[!] 동의어 로드 실패: {e}")
        SYNONYM_GROUPS = []
        SYNONYM_MAP = {}


def normalize_keyword(keyword: str) -> str:
    """키워드를 정규화된 형태로 변환합니다."""
    if not keyword:
        return ""
    normalized = keyword.lower().strip()
    return SYNONYM_MAP.get(normalized, normalized)


def are_synonyms(keyword1: str, keyword2: str) -> bool:
    """두 키워드가 동의어인지 확인합니다."""
    return normalize_keyword(keyword1) == normalize_keyword(keyword2)


# ==============================
# 3-2) 피드백 기반 점수 조정 로딩
# ==============================

def load_feedback_scores(path: Path = FEEDBACK_PATH, force_reload: bool = False):
    """
    피드백 데이터를 로드하여 문서별 점수 조정 가중치를 계산합니다.
    좋은 피드백을 받은 문서는 점수 증가, 나쁜 피드백은 감소.
    
    Args:
        path: 피드백 파일 경로
        force_reload: True이면 캐시를 무시하고 강제로 다시 로드
    """
    global FEEDBACK_SCORES, FEEDBACK_LAST_MODIFIED
    
    # DOCS가 아직 로드되지 않았으면 스킵
    if not DOCS:
        FEEDBACK_SCORES = {}
        return
    
    if not path.exists():
        if FEEDBACK_SCORES:
            QUERY_CACHE.clear()
        FEEDBACK_SCORES = {}
        FEEDBACK_LAST_MODIFIED = 0.0
        if force_reload:
            QUERY_CACHE.clear()
        return
    
    # 캐싱: 파일이 변경되지 않았고 강제 리로드가 아니면 스킵
    try:
        current_mtime = os.path.getmtime(path)
        cache_changed = force_reload or current_mtime != FEEDBACK_LAST_MODIFIED
        if not force_reload and FEEDBACK_SCORES and not cache_changed:
            return  # 캐시된 값 사용
        if cache_changed:
            QUERY_CACHE.clear()
        FEEDBACK_LAST_MODIFIED = current_mtime
    except OSError:
        QUERY_CACHE.clear()
        FEEDBACK_SCORES = {}
        return
    
    try:
        with path.open("r", encoding="utf-8") as f:
            feedback_data = json.load(f)
        
        feedbacks = feedback_data.get("feedbacks", [])
        if not feedbacks:
            FEEDBACK_SCORES = {}
            return
        
        # 문서 텍스트 -> 인덱스 매핑 (효율적인 조회를 위해)
        doc_text_to_idx = {}
        doc_id_to_idx: Dict[Any, List[int]] = {}
        for idx, doc_text in enumerate(DOCS):
            normalized_text = doc_text.strip()
            # 정규화된 텍스트로 매핑 (중복 방지)
            if normalized_text not in doc_text_to_idx:
                doc_text_to_idx[normalized_text] = []
            doc_text_to_idx[normalized_text].append(idx)
            if idx < len(DOC_ITEMS):
                doc_id = DOC_ITEMS[idx].get("id")
                if doc_id is not None:
                    doc_id_to_idx.setdefault(doc_id, []).append(idx)
        
        # 각 문서에 대한 피드백 점수 누적
        doc_feedback_scores = {}  # 인덱스 -> 누적 점수
        
        for feedback in feedbacks:
            if feedback.get("feedback") not in ["like", "dislike"]:
                continue
            
            contexts = feedback.get("contexts", [])
            feedback_type = feedback.get("feedback")
            
            # 좋은 피드백: +1, 나쁜 피드백: -1
            score_adjustment = 1.0 if feedback_type == "like" else -1.0
            
            # 각 컨텍스트에 대해 해당하는 문서 인덱스 찾기
            for ctx in contexts:
                doc_id = ctx.get("document_id", ctx.get("id"))
                if isinstance(doc_id, (int, str)) and doc_id in doc_id_to_idx:
                    for idx in doc_id_to_idx[doc_id]:
                        doc_feedback_scores[idx] = doc_feedback_scores.get(idx, 0.0) + score_adjustment
                    continue

                ctx_text = str(ctx.get("text", "")).strip()
                if not ctx_text:
                    continue
                
                # 효율적인 매칭: 정규화된 텍스트로 직접 조회
                ctx_normalized = ctx_text
                
                # 정확한 매칭 우선
                if ctx_normalized in doc_text_to_idx:
                    for idx in doc_text_to_idx[ctx_normalized]:
                        if idx not in doc_feedback_scores:
                            doc_feedback_scores[idx] = 0.0
                        doc_feedback_scores[idx] += score_adjustment
                else:
                    # 부분 매칭 (정확한 매칭이 없는 경우만)
                    for doc_text, idx_list in doc_text_to_idx.items():
                        if ctx_normalized in doc_text or doc_text in ctx_normalized:
                            for idx in idx_list:
                                if idx not in doc_feedback_scores:
                                    doc_feedback_scores[idx] = 0.0
                                doc_feedback_scores[idx] += score_adjustment * 0.5  # 부분 매칭은 가중치 0.5
        
        # 점수를 정규화 (최대 ±0.2 범위로 제한)
        if doc_feedback_scores:
            max_abs_score = max(abs(s) for s in doc_feedback_scores.values())
            if max_abs_score > 0:
                FEEDBACK_SCORES = {
                    idx: (score / max_abs_score) * 0.2  # 최대 ±0.2 보정
                    for idx, score in doc_feedback_scores.items()
                }
            else:
                FEEDBACK_SCORES = {}
            if force_reload:
                QUERY_CACHE.clear()
        else:
            FEEDBACK_SCORES = {}
        
    except Exception as e:
        print(f"[!] 피드백 점수 로드 실패: {e}")
        FEEDBACK_SCORES = {}


# ==============================
# 4) 키워드 통계
# ==============================

def get_keyword_stats() -> Dict[str, int]:
    _init_models_if_needed()
    stats: Dict[str, int] = {}
    for kw in DOC_KEYWORDS:
        stats[kw] = stats.get(kw, 0) + 1
    return stats


# ==============================
# 5) 검색 (Retrieval) - 하이브리드 검색
# ==============================

def extract_keywords_from_query(query: str) -> List[str]:
    """쿼리에서 키워드를 추출합니다."""
    query_lower = query.lower()
    found_keywords = []
    
    # 모든 키워드 확인
    for keyword in set(DOC_KEYWORDS):
        if keyword.lower() in query_lower:
            found_keywords.append(keyword)
    
    # 동의어도 확인
    for keyword in set(DOC_KEYWORDS):
        normalized_kw = normalize_keyword(keyword)
        # 쿼리의 단어들을 확인
        query_words = query_lower.split()
        for word in query_words:
            if normalize_keyword(word) == normalized_kw:
                if keyword not in found_keywords:
                    found_keywords.append(keyword)
    
    return found_keywords


def compute_keyword_score(doc_keyword: str, query: str) -> float:
    """문서 키워드와 쿼리 간의 키워드 매칭 점수를 계산합니다."""
    doc_keyword_lower = doc_keyword.lower()
    query_lower = query.lower()
    
    # 1. 정확한 키워드 매칭
    if doc_keyword_lower in query_lower:
        return 1.0
    
    # 2. 동의어 매칭
    query_words = query_lower.split()
    for word in query_words:
        if are_synonyms(word, doc_keyword):
            return 0.8
    
    # 3. 부분 매칭 (키워드가 쿼리에 포함)
    if doc_keyword_lower in query_lower or query_lower in doc_keyword_lower:
        return 0.5
    
    return 0.0


def retrieve_with_mmr(
    query: str,
    top_k: int = 3,
    filter_keyword: Optional[str] = None,
    hybrid_weight: float = 0.3,
    mmr_lambda: float = 0.5,  # MMR 파라미터 (0.0 = 다양성만, 1.0 = 관련성만)
    use_mmr: bool = True
):
    """
    MMR (Maximal Marginal Relevance) 기반 하이브리드 검색
    검색 결과의 다양성을 보장하여 중복된 컨텍스트를 제거합니다.
    
    Args:
        query: 검색 쿼리
        top_k: 상위 K개 결과 반환
        filter_keyword: 키워드 필터 (선택)
        hybrid_weight: 키워드 점수 가중치 (0.0-1.0)
        mmr_lambda: MMR 파라미터 (0.0-1.0), 높을수록 관련성 우선, 낮을수록 다양성 우선
        use_mmr: MMR 사용 여부
    """
    _init_models_if_needed()
    
    if not use_mmr or top_k <= 1:
        # MMR을 사용하지 않거나 top_k가 1 이하면 일반 검색
        return retrieve(query, top_k, filter_keyword, hybrid_weight)
    
    # 먼저 더 많은 후보를 검색 (top_k * 2)
    candidate_k = min(top_k * 3, 20)  # 최대 20개 후보
    candidates = retrieve(query, candidate_k, filter_keyword, hybrid_weight)
    
    if len(candidates) <= 1:
        return candidates
    
    # 쿼리 임베딩
    q_emb = SEMB_MODEL.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].astype("float32")
    
    # 후보 문서 임베딩
    candidate_embeds = []
    candidate_items = []
    for item in candidates:
        text = item[0]
        emb = SEMB_MODEL.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0].astype("float32")
        candidate_embeds.append(emb)
        candidate_items.append(item)
    
    candidate_embeds = np.array(candidate_embeds)
    
    # MMR 알고리즘
    selected = []
    selected_embeds = []
    remaining = list(range(len(candidates)))
    
    # 첫 번째 문서: 가장 관련성 높은 것 선택
    if remaining:
        first_idx = 0  # 이미 관련성 순으로 정렬되어 있음
        selected.append(first_idx)
        selected_embeds.append(candidate_embeds[first_idx])
        remaining.remove(first_idx)
    
    # 나머지 문서: MMR 점수로 선택
    while len(selected) < top_k and remaining:
        best_mmr_score = -float('inf')
        best_idx = None
        
        for idx in remaining:
            # 관련성 점수 (쿼리와의 유사도)
            relevance = float(np.dot(candidate_embeds[idx], q_emb))
            
            # 다양성 점수 (이미 선택된 문서들과의 최대 유사도)
            if selected_embeds:
                similarities = [float(np.dot(candidate_embeds[idx], sel_emb)) for sel_emb in selected_embeds]
                diversity = 1.0 - max(similarities)  # 최대 유사도가 낮을수록 다양성 높음
            else:
                diversity = 1.0
            
            # MMR 점수: 관련성과 다양성의 가중 평균
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * diversity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            selected_embeds.append(candidate_embeds[best_idx])
            remaining.remove(best_idx)
        else:
            break
    
    # 선택된 문서 반환 (원래 순서 유지)
    return [candidate_items[i] for i in selected]


def retrieve(
    query: str,
    top_k: int = 3,
    filter_keyword: Optional[str] = None,
    hybrid_weight: float = 0.3,  # 키워드 점수 가중치 (0.0 = 의미 검색만, 1.0 = 키워드 검색만)
    use_mmr: bool = False,  # MMR 사용 여부
    mmr_lambda: float = 0.5,  # MMR 파라미터
):
    """
    하이브리드 검색: 키워드 매칭 + 의미 기반 검색 (해시 테이블 최적화)
    
    Args:
        query: 검색 쿼리
        top_k: 상위 K개 결과 반환
        filter_keyword: 키워드 필터 (선택)
        hybrid_weight: 키워드 점수 가중치 (0.0-1.0)
        use_mmr: MMR 사용 여부 (기본: False)
        mmr_lambda: MMR 파라미터 (0.0-1.0)
    """
    if use_mmr:
        return retrieve_with_mmr(query, top_k, filter_keyword, hybrid_weight, mmr_lambda, True)
    _init_models_if_needed()

    # 1. 해시 테이블을 사용한 키워드 기반 후보 문서 필터링 (O(1) 조회)
    candidate_idxs = None
    
    # 쿼리에서 키워드 추출
    query_keywords = extract_keywords_from_query(query)
    
    if query_keywords or filter_keyword:
        candidate_idxs_set = set()
        
        # 쿼리 키워드로 후보 문서 찾기
        for kw in query_keywords:
            # 정확한 키워드 매칭
            if kw in KEYWORD_INDEX:
                candidate_idxs_set.update(KEYWORD_INDEX[kw])
            # 정규화된 키워드 매칭
            normalized_kw = normalize_keyword(kw)
            if normalized_kw in KEYWORD_INDEX_NORMALIZED:
                candidate_idxs_set.update(KEYWORD_INDEX_NORMALIZED[normalized_kw])
        
        # 필터 키워드 적용
        if filter_keyword:
            filter_idxs_set = set()
            # 정확한 키워드 매칭
            if filter_keyword in KEYWORD_INDEX:
                filter_idxs_set.update(KEYWORD_INDEX[filter_keyword])
            # 정규화된 키워드 매칭
            filter_normalized = normalize_keyword(filter_keyword)
            if filter_normalized in KEYWORD_INDEX_NORMALIZED:
                filter_idxs_set.update(KEYWORD_INDEX_NORMALIZED[filter_normalized])
            
            if candidate_idxs_set:
                # 교집합: 쿼리 키워드와 필터 키워드 모두 만족
                candidate_idxs_set = candidate_idxs_set & filter_idxs_set
            else:
                # 필터만 적용
                candidate_idxs_set = filter_idxs_set
        
        if candidate_idxs_set:
            candidate_idxs = list(candidate_idxs_set)
    
    # 필터 키워드만 있고 쿼리 키워드가 없는 경우
    if filter_keyword and not query_keywords and candidate_idxs is None:
        candidate_idxs_set = set()
        if filter_keyword in KEYWORD_INDEX:
            candidate_idxs_set.update(KEYWORD_INDEX[filter_keyword])
        filter_normalized = normalize_keyword(filter_keyword)
        if filter_normalized in KEYWORD_INDEX_NORMALIZED:
            candidate_idxs_set.update(KEYWORD_INDEX_NORMALIZED[filter_normalized])
        if candidate_idxs_set:
            candidate_idxs = list(candidate_idxs_set)
    
    # 해당 keyword 문서가 없는 경우
    if (filter_keyword or query_keywords) and (candidate_idxs is None or len(candidate_idxs) == 0):
        return []
    
    # 검색 대상 문서 인덱스 결정
    search_idxs = candidate_idxs if candidate_idxs is not None else list(range(len(DOCS)))

    # 2. 의미 기반 검색 (후보 문서에 대해서만 수행)
    q_emb = SEMB_MODEL.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].astype("float32")

    # 후보 문서의 임베딩만 사용
    candidate_embeds = DOC_EMBEDS[search_idxs]
    semantic_scores_candidate = np.dot(candidate_embeds, q_emb)
    
    # 전체 문서에 대한 점수 배열 생성
    semantic_scores = np.full(len(DOCS), -1.0, dtype=np.float32)
    for i, idx in enumerate(search_idxs):
        semantic_scores[idx] = semantic_scores_candidate[i]
    
    # 정규화: 0-1 범위로 변환 (후보 문서만 고려)
    valid_scores = semantic_scores[search_idxs]
    semantic_min, semantic_max = valid_scores.min(), valid_scores.max()
    if semantic_max > semantic_min:
        semantic_scores_normalized = np.full(len(DOCS), 0.0, dtype=np.float32)
        for idx in search_idxs:
            semantic_scores_normalized[idx] = (semantic_scores[idx] - semantic_min) / (semantic_max - semantic_min)
    else:
        semantic_scores_normalized = np.full(len(DOCS), 0.5, dtype=np.float32)

    # 3. 키워드 매칭 점수 계산 (후보 문서에 대해서만)
    keyword_scores = np.zeros(len(DOCS), dtype=np.float32)
    for idx in search_idxs:
        keyword_scores[idx] = compute_keyword_score(DOC_KEYWORDS[idx], query)

    # 4. 하이브리드 점수 결합
    hybrid_scores = (1 - hybrid_weight) * semantic_scores_normalized + hybrid_weight * keyword_scores
    
    # 5. 피드백 기반 점수 조정 적용
    load_feedback_scores()  # 최신 피드백 반영
    for idx in search_idxs:
        if idx in FEEDBACK_SCORES:
            hybrid_scores[idx] += FEEDBACK_SCORES[idx]
            # 점수를 0-1 범위로 클리핑
            hybrid_scores[idx] = max(0.0, min(1.0, hybrid_scores[idx]))

    # 6. 상위 K 선택 (후보 문서 중에서만)
    sorted_idx = sorted(search_idxs, key=lambda i: -hybrid_scores[i])[:top_k]

    return [
        (
            DOCS[i], 
            float(hybrid_scores[i]), 
            DOC_KEYWORDS[i],
            DOC_ITEMS[i].get("id") if i < len(DOC_ITEMS) else None  # 문서 ID 추가
        )
        for i in sorted_idx
    ]


def debug_retrieval(query: str, retrieved):
    print("\n[🔍 검색 디버그]")
    print(f"- 질문: {query}")

    if not retrieved:
        print("  (검색 결과 없음)")
        return

    scores = [s for _, s, _, _ in retrieved] if len(retrieved) > 0 and len(retrieved[0]) == 4 else [s for _, s, _ in retrieved]
    print(f"  * score range: min={min(scores):.3f}, max={max(scores):.3f}")

    for item in retrieved:
        if len(item) == 4:
            text, score, keyword, doc_id = item
            print(f"    - [ID:{doc_id}][{keyword}] score={score:.3f} | {text}")
        else:
            text, score, keyword = item
            print(f"    - [{keyword}] score={score:.3f} | {text}")


# ==============================
# 5) 쿼리 캐싱 유틸리티
# ==============================

def _generate_with_semantic_routing(
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    """Let the small model judge context answerability and answer simple questions."""
    router_instruction = (
        "먼저 질문과 제공된 문맥만 보고 답변 난이도를 판단하세요. "
        "문맥의 사실을 그대로 찾거나 간단히 조합해 답할 수 있으면 SIMPLE로 분류하고, "
        "문맥에 설명된 두 장비의 차이를 요약하는 단순 비교도 SIMPLE입니다. "
        "원인과 결과를 여러 단계로 추론하거나, 여러 요소를 종합해 진단·최적화해야 하면 COMPLEX입니다. "
        "단어 '차이'나 '비교'만으로 COMPLEX라고 판단하지 마세요. "
        "출력 형식은 반드시 지키세요: 간단한 질문은 첫 줄에 SIMPLE, 다음 줄부터 문맥에 근거한 한국어 답변. "
        "복잡한 질문은 COMPLEX 한 단어만 출력하세요. 분류 이유는 출력하지 마세요."
    )
    routed_messages = [dict(message) for message in messages]
    routed_messages[0]["content"] = router_instruction + "\n\n" + routed_messages[0]["content"]
    routed_output = generate_text(
        routed_messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    ).strip()

    route_match = re.match(r"^(SIMPLE|COMPLEX)\b\s*", routed_output, re.IGNORECASE)
    if route_match and route_match.group(1).upper() == "SIMPLE":
        answer = routed_output[route_match.end():].strip()
        if answer:
            return answer, "simple: context answerable"

    # If the route label is missing or invalid, use 7B instead of trusting a guess.
    return generate_text(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        force_local_model=True,
    ), "complex: requires deeper reasoning"

def _get_cache_key(
    query: str,
    top_k: int,
    filter_keyword: Optional[str],
    max_new_tokens: int,
    model_route: str = "default",
) -> str:
    """Build a cache key that includes every setting affecting the generated answer."""
    key_str = (
        f"{query}|{top_k}|{filter_keyword or ''}|{max_new_tokens}|"
        f"{LLM_BASE_URL}|{LLM_REMOTE_MODEL if LLM_BASE_URL else LLM_MODEL_NAME}|{model_route}"
    )
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def _get_from_cache(cache_key: str) -> Optional[Tuple[str, List[Tuple[str, float, str, Optional[int]]], float, List[str]]]:
    """캐시에서 결과 가져오기"""
    if cache_key in QUERY_CACHE:
        # LRU: 사용된 항목을 맨 뒤로 이동
        result = QUERY_CACHE.pop(cache_key)
        QUERY_CACHE[cache_key] = result
        # 호환성을 위해 관련 질문이 없으면 빈 리스트 추가
        if len(result) == 3:
            return (*result, [])
        return result
    return None


def _save_to_cache(
    cache_key: str,
    answer: str,
    retrieved: List,
    confidence: float,
    related_questions: Optional[List[str]] = None,
):
    """캐시에 결과 저장"""
    # 캐시 크기 제한
    if len(QUERY_CACHE) >= MAX_CACHE_SIZE:
        # 가장 오래된 항목 제거 (FIFO)
        QUERY_CACHE.popitem(last=False)
    
    QUERY_CACHE[cache_key] = (answer, retrieved, confidence, related_questions or [])


def calculate_confidence_score(retrieved: List) -> float:
    """검색 결과 기반 신뢰도 점수 계산 (0.0 ~ 1.0)"""
    if not retrieved:
        return 0.0
    
    # retrieved는 (text, score, keyword, doc_id) 튜플 리스트
    scores = []
    for item in retrieved:
        if len(item) >= 2:
            scores.append(item[1])  # score
    
    if not scores:
        return 0.0
    
    # 평균 점수와 최고 점수를 고려
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    
    # 가중 평균: 최고 점수 70%, 평균 점수 30%
    confidence = 0.7 * max_score + 0.3 * avg_score
    
    # 점수를 0.0 ~ 1.0 범위로 정규화 (이미 0~1 범위일 가능성이 높지만 안전장치)
    return min(1.0, max(0.0, confidence))


def generate_related_questions(query: str, retrieved: List, max_questions: int = 3) -> List[str]:
    """검색 결과 기반으로 관련 질문 생성"""
    if not retrieved or len(retrieved) == 0:
        return []
    
    # 검색된 키워드 수집
    keywords = set()
    for item in retrieved:
        if len(item) >= 3:
            keywords.add(item[2])  # keyword
    
    if not keywords:
        return []
    
    # 키워드 조합 기반 질문 템플릿
    keyword_list = list(keywords)[:5]  # 최대 5개 키워드만 사용
    
    # 간단한 질문 템플릿 생성
    questions = []
    
    # 단일 키워드 질문
    for kw in keyword_list[:2]:
        questions.append(f"{kw}는 무엇인가요?")
        questions.append(f"{kw}에 대해 설명해주세요.")
    
    # 키워드 조합 질문
    if len(keyword_list) >= 2:
        questions.append(f"{keyword_list[0]}와 {keyword_list[1]}의 관계는 무엇인가요?")
    
    if len(keyword_list) >= 3:
        questions.append(f"{keyword_list[0]}, {keyword_list[1]}, {keyword_list[2]}에 대해 알려주세요.")
    
    # 원본 질문과 유사한 패턴의 질문 생성
    if "어떻게" in query or "방법" in query:
        for kw in keyword_list[:2]:
            questions.append(f"{kw}는 어떻게 사용하나요?")
    elif "무엇" in query or "뭐" in query:
        for kw in keyword_list[:2]:
            questions.append(f"{kw}의 역할은 무엇인가요?")
    
    # 중복 제거 및 최대 개수 제한
    unique_questions = []
    seen = set()
    for q in questions:
        if q not in seen and q != query:
            seen.add(q)
            unique_questions.append(q)
            if len(unique_questions) >= max_questions:
                break
    
    return unique_questions


# ==============================
# 6) Qwen 기반 RAG 생성 (개선된 버전)
# ==============================

def generate_answer(
    query: str,
    top_k: int = 3,
    max_new_tokens: int = 256,
    filter_keyword: Optional[str] = None,
    context_only: bool = False,
    debug: bool = False,
    use_cache: bool = True,
):

    _init_models_if_needed()
    
    # The small model evaluates answerability from the retrieved evidence.
    # Without a configured endpoint, generation uses local 7B QLoRA.
    model_route = f"semantic_router:{LLM_REMOTE_MODEL}" if LLM_BASE_URL else "local_qwen7b_qlora"
    # 캐시 확인 (context_only 모드가 아니고 캐시 사용이 활성화된 경우)
    cache_key = None
    if use_cache and not context_only:
        cache_key = _get_cache_key(query, top_k, filter_keyword, max_new_tokens, model_route)
        cached_result = _get_from_cache(cache_key)
        if cached_result is not None:
            if len(cached_result) == 4:
                answer, retrieved, confidence, related_questions = cached_result
            else:
                answer, retrieved, confidence = cached_result[:3]
                related_questions = []
            if debug:
                print(f"[캐시 히트] 쿼리: {query[:50]}...")
            return answer, retrieved, confidence, related_questions

    # 검색 수행 (MMR 기본 활성화)
    retrieved = retrieve(query, top_k=top_k, filter_keyword=filter_keyword, use_mmr=True, mmr_lambda=0.6)

    if debug:
        debug_retrieval(query, retrieved)

    if not retrieved:
        return (
            "해당 질문과 일치하는 문맥을 찾지 못했어.\n"
            "→ filter_keyword가 너무 좁거나\n"
            "→ docs_ald.json에 관련 문장이 부족할 수 있어.",
            [],
            0.0,
            []
        )

    # retrieved는 (text, score, keyword, doc_id) 튜플 리스트
    scores = [item[1] for item in retrieved if len(item) >= 2]
    max_score = max(scores) if scores else 0.0

    # 안전장치
    if max_score < 0.45:  
        confidence = calculate_confidence_score(retrieved)
        related_questions = generate_related_questions(query, retrieved, max_questions=3)
        return (
            "문맥과의 연관성이 너무 낮아서 답변을 생성하지 않았어.\n"
            "문서를 보강하거나 질문을 더 구체적으로 바꿔줘!",
            retrieved,
            confidence,
            related_questions
        )

    # context만 반환 모드
    if context_only:
        confidence = calculate_confidence_score(retrieved)
        related_questions = generate_related_questions(query, retrieved, max_questions=3)
        return "컨텍스트만 반환했어.", retrieved, confidence, related_questions

    # LLM 프롬프트 구성 (문서 ID 포함)
    ctx_parts = []
    for item in retrieved:
        if len(item) == 4:
            text, score, kw, doc_id = item
            doc_id_str = f"[문서 ID: {doc_id}]" if doc_id else ""
            ctx_parts.append(f"- {doc_id_str} ({kw}) {text}")
        else:
            text, score, kw = item[:3]
            ctx_parts.append(f"- ({kw}) {text}")
    ctx = "\n".join(ctx_parts)

    system_prompt = (
        "반도체 ALD 전문가로서 답변하세요.\n"
        "문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n"
        "반드시 한국어로만 답변해야 합니다.\n"
        "답변 마지막에 참조한 문서 ID를 [참조: 문서 ID1, 문서 ID2, ...] 형식으로 표시하세요."
    )

    user_prompt = f"""
아래는 관련 문맥이야:

{ctx}

[질문]
{query}

위 문맥만 근거로 한국어로 답변해줘.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if LLM_BASE_URL:
        answer, route_reason = _generate_with_semantic_routing(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
        )
        if route_reason.startswith("simple:"):
            print(f"[LLM Router] simple -> {LLM_REMOTE_MODEL} ({route_reason})")
        else:
            print(f"[LLM Router] complex -> local Qwen 7B QLoRA ({route_reason})")
    else:
        print("[LLM Router] local Qwen 7B QLoRA (no small-model endpoint configured)")
        answer = generate_text(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            force_local_model=True,
        )
    # 답변에 소스 인용 추가 (LLM이 추가하지 않은 경우)
    doc_ids = [item[3] for item in retrieved if len(item) == 4 and item[3] is not None]
    if doc_ids and "[참조:" not in answer and "[문서 ID:" not in answer:
        unique_doc_ids = sorted(set(doc_ids))
        answer += f"\n\n[참조: 문서 ID {', '.join(map(str, unique_doc_ids))}]"
    
    # 신뢰도 점수 계산
    confidence = calculate_confidence_score(retrieved)
    
    # 관련 질문 생성
    related_questions = generate_related_questions(query, retrieved, max_questions=3)
    
    # 캐시에 저장
    if cache_key is not None:
        _save_to_cache(cache_key, answer, retrieved, confidence, related_questions)

    return answer, retrieved, confidence, related_questions


def generate_text(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
    force_local_model: bool = False,
) -> str:
    """Generate through a configured OpenAI-compatible server or the local dev model."""
    if LLM_BASE_URL and not force_local_model:
        import httpx  # type: ignore

        headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
        payload = {
            "model": LLM_REMOTE_MODEL,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if "127.0.0.1:11434" in LLM_BASE_URL or "localhost:11434" in LLM_BASE_URL:
            payload["keep_alive"] = 0
        response = httpx.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("LLM endpoint returned an invalid chat-completions response") from error
        if not isinstance(content, str):
            raise ValueError("LLM endpoint returned non-text message content")
        return content.strip()

    _init_models_if_needed()
    _load_local_llm_if_needed()
    if LLM is None or TOKENIZER is None:
        raise RuntimeError("Local LLM was not initialized")

    import torch  # type: ignore

    prompt = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = TOKENIZER(prompt, return_tensors="pt")
    if hasattr(LLM, "device"):
        model_device = LLM.device
    else:
        model_device = next(LLM.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = LLM.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return TOKENIZER.decode(generated_ids, skip_special_tokens=True).strip()