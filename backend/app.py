import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from rag_core import (
    generate_answer,
    generate_text,
    get_keyword_stats,
    load_feedback_scores,
    reload_documents,
)
from rule_engine import RouteResult, route_question

BASE_DIR = Path(__file__).resolve().parent.parent
FEEDBACK_PATH = BASE_DIR / "feedback" / "feedback_data.json"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALD_RAG_CORS_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if origin.strip()
]
logger = logging.getLogger(__name__)


# ============================================
# FastAPI 기본 세팅
# ============================================

app = FastAPI(
    title="반도체 ALD RAG API",
    description="반도체 지식 기반 ALD RAG API",
    version="1.4.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request / Response 모델
# ============================================

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    filter_keyword: Optional[str] = Field(default=None, max_length=100)
    context_only: bool = False
    debug: bool = False


class ContextItem(BaseModel):
    text: str
    score: float
    keyword: str
    document_id: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    contexts: List[ContextItem]
    used_keyword: str
    session_id: Optional[str] = None  # 피드백 추적용 세션 ID
    confidence: Optional[float] = None  # 답변 신뢰도 점수 (0.0 ~ 1.0)
    related_questions: Optional[List[str]] = None  # 관련 질문 목록
    answer_type: str = "knowledge_qa"
    rule_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=10000)
    contexts: List[ContextItem] = Field(max_length=20)
    feedback: Literal["like", "dislike"]
    comment: Optional[str] = Field(default=None, max_length=2000)


# ============================================
# 기본 건강 체크 엔드포인트
# ============================================

@app.get("/")
def root():
    return {
        "message": "ALD RAG API 정상 동작 중",
        "keywords": {},
        "num_docs": 0,
        "keyword_list": [],
        "note": "POST /chat 으로 질문할 수 있습니다.",
        "endpoints": {
            "chat": "POST /chat - 질문하기",
            "keywords": "GET /keywords - 키워드 통계",
            "health": "GET /health/ready - 모델 준비 상태",
            "docs": "GET /docs - Swagger UI",
        },
    }


@app.get("/health/ready")
def readiness():
    """Load/check retrieval dependencies without coupling API liveness to model startup."""
    from rag_core import MODEL_INFO, _init_models_if_needed

    try:
        _init_models_if_needed()
    except Exception as error:
        logger.exception("RAG readiness check failed")
        raise HTTPException(status_code=503, detail="RAG dependencies are not ready") from error

    return {
        "status": "ready",
        "num_docs": MODEL_INFO.get("num_docs", 0),
        "llm_backend": MODEL_INFO.get("llm_backend", "uninitialized"),
    }


# ============================================
# 핵심: /chat RAG 엔드포인트
# ============================================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Apply deterministic safety/rule handling before invoking RAG."""
    route = route_question(req.question)
    if route.answer is not None:
        return ChatResponse(
            answer=route.answer,
            contexts=[],
            used_keyword=req.filter_keyword or "",
            session_id=uuid.uuid4().hex,
            confidence=1.0,
            related_questions=[],
            answer_type=route.route,
            rule_id=route.rule_id,
        )
    return _chat_rag(req, route)


def _chat_rag(req: ChatRequest, route: RouteResult) -> ChatResponse:
    """Run retrieval/generation only after rule routing has allowed the request."""
    try:
        result = generate_answer(
            query=req.question,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            filter_keyword=req.filter_keyword,
            context_only=req.context_only,
            debug=req.debug
        )
        
        if len(result) == 4:
            answer, retrieved, confidence, related_questions = result
        elif len(result) == 3:
            answer, retrieved, confidence = result
            related_questions = []
        elif len(result) == 2:
            answer, retrieved = result
            confidence = 0.0
            related_questions = []
        else:
            raise ValueError("generate_answer returned an unsupported result shape")
    except Exception as error:
        logger.exception("RAG answer generation failed")
        raise HTTPException(status_code=500, detail="답변 생성에 실패했습니다.") from error

    retrieved = retrieved or []

    context_items: List[ContextItem] = []
    for item in retrieved:
        try:
            document_id = None
            if isinstance(item, tuple):
                if len(item) >= 4:
                    text, score, keyword, document_id = item[:4]
                elif len(item) >= 3:
                    text, score, keyword = item[:3]
                else:
                    text = str(item[0]) if len(item) > 0 else ""
                    score = float(item[1]) if len(item) > 1 else 0.0
                    keyword = ""
            elif isinstance(item, dict):
                text = item.get("text", "")
                score = float(item.get("score", 0.0))
                keyword = item.get("keyword", "")
                document_id = item.get("document_id", item.get("id"))
            else:
                raise TypeError(f"Unsupported retrieved context type: {type(item).__name__}")

            context_items.append(ContextItem(
                text=str(text),
                score=float(score),
                keyword=str(keyword) if keyword else "",
                document_id=document_id,
            ))
        except (TypeError, ValueError, IndexError) as error:
            logger.warning("Ignoring malformed retrieval context: %s", error)

    used_keyword = req.filter_keyword or ""
    if not used_keyword and context_items:
        # 가장 많이 나타난 키워드 사용
        keyword_counts = {}
        for ctx in context_items:
            if ctx.keyword:
                keyword_counts[ctx.keyword] = keyword_counts.get(ctx.keyword, 0) + 1
        if keyword_counts:
            used_keyword = max(keyword_counts.items(), key=lambda x: x[1])[0]

    return ChatResponse(
        answer=answer or "답변이 생성되지 않았습니다.",
        contexts=context_items,
        used_keyword=used_keyword,
        session_id=uuid.uuid4().hex,
        confidence=confidence,
        related_questions=related_questions,
        answer_type=route.route,
        rule_id=route.rule_id,
    )


# ============================================
# 피드백 수집 엔드포인트
# ============================================

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    사용자 피드백 수집 (👍/👎)
    """
    try:
        # 피드백 데이터 로드
        if FEEDBACK_PATH.exists():
            with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        else:
            feedback_data = {"feedbacks": []}
        
        # 새 피드백 추가
        feedback_entry = {
            "session_id": req.session_id,
            "question": req.question,
            "answer": req.answer,
            "contexts": [
                {
                    "text": ctx.text,
                    "score": ctx.score,
                    "keyword": ctx.keyword,
                    "document_id": ctx.document_id,
                }
                for ctx in req.contexts
            ],
            "feedback": req.feedback,  # "like" or "dislike"
            "comment": req.comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        feedback_data["feedbacks"].append(feedback_entry)
        
        # 파일에 저장
        with FEEDBACK_PATH.open("w", encoding="utf-8") as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        # 피드백 기반 점수 조정 갱신 (강제 리로드)
        load_feedback_scores(force_reload=True)
        
        print(f"[+] 피드백 수집: {req.feedback} (session_id: {req.session_id})")
        print(f"[+] 피드백이 저장되었고, 검색 점수가 업데이트되었습니다.")
        
        return {
            "success": True,
            "message": "피드백이 저장되었고, 다음 검색부터 반영됩니다.",
            "session_id": req.session_id
        }
    except Exception as e:
        import traceback
        print(f"[ERROR] 피드백 저장 실패: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/feedback/stats")
def get_feedback_stats():
    """피드백 통계 조회"""
    try:
        if not FEEDBACK_PATH.exists():
            return {
                "total": 0,
                "likes": 0,
                "dislikes": 0,
                "recent_feedbacks": []
            }
        
        with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
            feedback_data = json.load(f)
        
        feedbacks = feedback_data.get("feedbacks", [])
        likes = sum(1 for fb in feedbacks if fb.get("feedback") == "like")
        dislikes = sum(1 for fb in feedbacks if fb.get("feedback") == "dislike")
        
        # 최근 10개 피드백
        recent = sorted(feedbacks, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]
        
        return {
            "total": len(feedbacks),
            "likes": likes,
            "dislikes": dislikes,
            "recent_feedbacks": recent
        }
    except Exception as e:
        return {
            "error": str(e)
        }


# ============================================
# 전체 키워드 목록 제공 (프론트에서 Select 박스 용)
# ============================================

@app.get("/keywords")
def keywords():
    """
    키워드별 문서 개수 통계 반환
    
    프론트엔드에서 키워드 필터 선택 박스를 채우는 데 사용됩니다.
    """
    try:
        stats = get_keyword_stats()
        return stats if stats else {}
    except Exception as e:
        print(f"[ERROR] get_keyword_stats() 실패: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================
# 문서 재로드 엔드포인트 (docs_ald.json 변경 후 사용)
# ============================================

@app.post("/reload-docs")
def reload_docs():
    """
    docs_ald.json 파일을 다시 로드하고 임베딩을 재생성합니다.
    
    docs_ald.json 파일을 수정한 후 이 엔드포인트를 호출하면
    서버를 재시작하지 않고도 새로운 문서가 검색에 반영됩니다.
    """
    try:
        num_docs = reload_documents()
        stats = get_keyword_stats()
        return {
            "success": True,
            "message": f"문서 재로딩 완료: {num_docs}개 문서",
            "num_docs": num_docs,
            "keywords": stats
        }
    except Exception as e:
        print(f"[ERROR] reload_documents() 실패: {e}")
        import traceback
        error_detail = traceback.format_exc()
        print(error_detail)
        return {
            "success": False,
            "error": str(e),
            "detail": error_detail
        }


# ============================================
# 문서 관리 엔드포인트들
# ============================================

@app.get("/docs/stats")
def docs_stats():
    """키워드별 문서 개수 통계"""
    try:
        # 직접 JSON 파일 읽기 (리스트 형태로)
        docs_path = BASE_DIR / "docs" / "docs_ald.json"
        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        docs = data.get("documents", [])
        if not isinstance(docs, list):
            docs = []
        
        keyword_counts = {}
        
        for item in docs:
            if not isinstance(item, dict):
                continue
            keywords = item.get("keywords", [])
            if isinstance(keywords, list):
                for kw in keywords:
                    kw = str(kw).strip()
                    if kw:
                        keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            elif isinstance(keywords, str):
                kw = keywords.strip()
                if kw:
                    keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        return {
            "success": True,
            "stats": keyword_counts,
            "total_docs": len(docs)
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "detail": traceback.format_exc()
        }


@app.get("/docs/group")
def docs_group():
    """키워드별로 그룹화된 문서 목록"""
    try:
        # 직접 JSON 파일 읽기 (리스트 형태로)
        docs_path = BASE_DIR / "docs" / "docs_ald.json"
        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        docs = data.get("documents", [])
        if not isinstance(docs, list):
            docs = []
        
        grouped = {}
        
        for item in docs:
            if not isinstance(item, dict):
                continue
            keywords = item.get("keywords", [])
            if isinstance(keywords, list):
                for kw in keywords:
                    kw = str(kw).strip()
                    if kw:
                        if kw not in grouped:
                            grouped[kw] = []
                        grouped[kw].append({
                            "id": item.get("id"),
                            "text": item.get("text", ""),
                            "keywords": item.get("keywords", [])
                        })
            elif isinstance(keywords, str):
                kw = keywords.strip()
                if kw:
                    if kw not in grouped:
                        grouped[kw] = []
                    grouped[kw].append({
                        "id": item.get("id"),
                        "text": item.get("text", ""),
                        "keywords": [keywords]
                    })
        
        return {
            "success": True,
            "grouped": grouped,
            "total_docs": len(docs),
            "total_keywords": len(grouped)
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "detail": traceback.format_exc()
        }


@app.post("/docs/add")
def docs_add(keyword: str = Form(...), text: str = Form(...)):
    """새 문서 추가"""
    try:
        # 직접 JSON 파일 읽기
        docs_path = BASE_DIR / "docs" / "docs_ald.json"
        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        docs = data.get("documents", [])
        if not isinstance(docs, list):
            docs = []
        
        keywords_list = [kw.strip() for kw in keyword.split(",") if kw.strip()]
        
        # 다음 ID 계산
        next_id = 1
        if docs:
            max_id = max((item.get("id", 0) for item in docs if isinstance(item, dict)), default=0)
            next_id = max_id + 1
        
        new_item = {
            "id": next_id,
            "keywords": keywords_list,
            "text": text
        }
        docs.append(new_item)
        
        # 파일 저장
        data["documents"] = docs
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 문서 재로드
        reload_documents()
        
        return {
            "success": True,
            "message": "문서가 추가되었습니다",
            "item": new_item
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "detail": traceback.format_exc()
        }


@app.post("/docs/extract")
async def docs_extract(
    file: UploadFile = File(...),
    keywords: str = Form(...),
    file_type: str = Form("text")
):
    """텍스트/PDF 파일에서 키워드 관련 문장 추출"""
    try:
        from scripts.doc_management.extract_from_docs import extract_from_text, extract_from_pdf, filter_quality_sentences
        import tempfile
        
        keywords_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        
        # 파일 읽기
        content = await file.read()
        
        if file_type == "pdf":
            # PDF 처리
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            
            try:
                results = extract_from_pdf(tmp_path, keywords_list)
            finally:
                tmp_path.unlink()
        else:
            # 텍스트 처리
            text = content.decode("utf-8")
            results = extract_from_text(text, keywords_list)
        
        # 품질 필터링
        for keyword in keywords_list:
            results[keyword] = filter_quality_sentences(results.get(keyword, []))
        
        # 중복 제거 및 문서 추가
        docs_path = BASE_DIR / "docs" / "docs_ald.json"
        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        docs = data.get("documents", [])
        if not isinstance(docs, list):
            docs = []
        
        existing_texts = {item.get("text", "") for item in docs if isinstance(item, dict)}
        new_count = 0
        
        # 다음 ID 계산
        next_id = 1
        if docs:
            max_id = max((item.get("id", 0) for item in docs if isinstance(item, dict)), default=0)
            next_id = max_id + 1
        
        extracted_items = []
        for keyword in keywords_list:
            sentences = results.get(keyword, [])
            for sentence in sentences:
                if sentence not in existing_texts:
                    new_item = {
                        "id": next_id,
                        "keywords": [keyword],
                        "text": sentence
                    }
                    docs.append(new_item)
                    extracted_items.append(new_item)
                    existing_texts.add(sentence)
                    next_id += 1
                    new_count += 1
        
        if new_count > 0:
            # 파일 저장
            data["documents"] = docs
            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            reload_documents()
        
        return {
            "success": True,
            "message": f"{new_count}개 문장이 추출되어 추가되었습니다",
            "extracted": {kw: len(results.get(kw, [])) for kw in keywords_list},
            "added": new_count,
            "items": extracted_items[:10]  # 최대 10개만 반환
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "detail": traceback.format_exc()
        }


@app.post("/docs/generate")
def docs_generate(
    mode: str = Form(...),
    keyword: str = Form(...),
    count: int = Form(3)
):
    """LLM 또는 템플릿을 사용하여 문서 생성"""
    try:
        from scripts.doc_management.generate_docs import generate_from_template
        
        if mode == "llm":
            # 직접 JSON 파일 읽기
            docs_path = BASE_DIR / "docs" / "docs_ald.json"
            with open(docs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            docs = data.get("documents", [])
            if not isinstance(docs, list):
                docs = []
            
            existing_texts = []
            for item in docs:
                if not isinstance(item, dict):
                    continue
                keywords = item.get("keywords", [])
                if isinstance(keywords, list) and keyword in keywords:
                    existing_texts.append(item.get("text", ""))
                elif isinstance(keywords, str) and keywords == keyword:
                    existing_texts.append(item.get("text", ""))
            
            if not existing_texts:
                return {
                    "success": False,
                    "error": f"'{keyword}' 키워드에 기존 데이터가 없습니다"
                }
            
            # LLM으로 생성 (간단한 버전)
            context_examples = "\n".join([f"- {text}" for text in existing_texts[:3]])
            system_prompt = (
                "반도체 ALD 전문가로서 답변하세요.\n"
                "문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n"
                "주어진 키워드와 관련된 짧고 명확한 설명 문장을 생성해야 합니다.\n"
                "반드시 한국어로 작성하고, 영어 전문 용어는 그대로 사용해도 됩니다.\n"
                "각 문장은 독립적이고, 실제 공정에서 사용되는 구체적인 정보를 담아야 합니다."
            )
            user_prompt = f"""
키워드: {keyword}

기존 예시 문장들:
{context_examples}

위 키워드와 관련된 새로운 설명 문장을 {count}개 생성해줘.
각 문장은:
- 50자 이내로 간결하게
- 실제 ALD 공정에서 사용되는 구체적인 정보
- 기존 예시와 유사한 스타일

생성된 문장만 한 줄에 하나씩 출력해줘 (번호나 설명 없이).
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            answer = generate_text(
                messages,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
            )
            
            lines = [line.strip() for line in answer.strip().split("\n") if line.strip()]
            cleaned_lines = []
            for line in lines:
                for prefix in ["1.", "2.", "3.", "4.", "5.", "- ", "* ", "• "]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line and len(line) > 10:
                    cleaned_lines.append(line)
            
            new_texts = cleaned_lines[:count]
            
            if not new_texts:
                return {
                    "success": False,
                    "error": "생성된 텍스트가 없습니다"
                }
            
            # 문서에 추가
            # 다음 ID 계산
            next_id = 1
            if docs:
                max_id = max((item.get("id", 0) for item in docs if isinstance(item, dict)), default=0)
                next_id = max_id + 1
            
            new_items = []
            for text in new_texts:
                new_item = {
                    "id": next_id,
                    "keywords": [keyword],
                    "text": text
                }
                docs.append(new_item)
                new_items.append(new_item)
                next_id += 1
            
            # 파일 저장
            data["documents"] = docs
            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            reload_documents()
            
            return {
                "success": True,
                "message": f"{len(new_texts)}개 문장이 생성되어 추가되었습니다",
                "items": new_items,
                "warning": "⚠️ 생성된 문장은 반드시 전문가 검토가 필요합니다"
            }
            
        elif mode == "template":
            new_texts = generate_from_template(keyword, count)
            
            if not new_texts:
                return {
                    "success": False,
                    "error": f"'{keyword}'에 대한 템플릿이 없습니다"
                }
            
            # 직접 JSON 파일 읽기
            docs_path = BASE_DIR / "docs" / "docs_ald.json"
            with open(docs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            docs = data.get("documents", [])
            if not isinstance(docs, list):
                docs = []
            
            # 다음 ID 계산
            next_id = 1
            if docs:
                max_id = max((item.get("id", 0) for item in docs if isinstance(item, dict)), default=0)
                next_id = max_id + 1
            
            new_items = []
            for text in new_texts:
                new_item = {
                    "id": next_id,
                    "keywords": [keyword],
                    "text": text
                }
                docs.append(new_item)
                new_items.append(new_item)
                next_id += 1
            
            # 파일 저장
            data["documents"] = docs
            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            reload_documents()
            
            return {
                "success": True,
                "message": f"{len(new_texts)}개 문장이 생성되어 추가되었습니다",
                "items": new_items
            }
        else:
            return {
                "success": False,
                "error": f"알 수 없는 모드: {mode}"
            }
            
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "detail": traceback.format_exc()
        }
