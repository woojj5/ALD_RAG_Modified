# rag_llama.py
"""
터미널에서 RAG 챗봇을 테스트하기 위한 CLI 스크립트.

- chat  모드: 질문 반복해서 입력하면서 대화
- once  모드: 한 번만 질문하고 종료 (셸 스크립트에서 쓰기 좋음)
- stats 모드: docs_ald.json 기준 키워드 통계 확인

사용 예시:
    cd ~/ald-rag-lab
    python rag_llama.py --mode chat
    python rag_llama.py --mode once -q "ALD에서 purge는 왜 필요한데?"
    python rag_llama.py --mode stats

추가 옵션:
    --filter-keyword "Precursor"
    --context-only
    --debug
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# rag_core import를 위해 경로 추가
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ~/ald-rag-lab
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag_core  # rag_core 모듈 전체 import

# rag_core에서 필요한 함수/정보 꺼내 쓰기
generate_answer = rag_core.generate_answer
get_keyword_stats = getattr(rag_core, "get_keyword_stats", None)
MODEL_INFO: Dict[str, Any] = getattr(rag_core, "MODEL_INFO", {})

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ==============================
# 1) 유틸 함수들
# ==============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="반도체 ALD RAG CLI")

    parser.add_argument(
        "--mode",
        choices=["chat", "once", "stats"],
        default="chat",
        help="chat: 대화 모드 / once: 한 번만 답변 / stats: 키워드 통계",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default="",
        help="once 모드에서 사용할 질문 (없으면 stdin에서 입력)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="검색할 상위 문장 개수",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="LLM이 생성할 최대 토큰 수",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="대화 로그를 저장할 파일 경로 (기본: logs/session_날짜.txt)",
    )
    parser.add_argument(
        "--filter-keyword",
        type=str,
        default="",
        help='특정 키워드만 검색 (예: "Precursor", "Purge", "Plasma", "MFC", "Chamber Pressure", "Flow")',
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="답변 생성 없이, 검색된 컨텍스트만 보고 싶을 때",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="검색 디버그 로그 출력 (유사도 범위 등)",
    )

    return parser.parse_args()


def init_log_file(path_str: str) -> Path:
    if path_str:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = LOG_DIR / f"session_{now}.txt"
    return p


def log_write(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def format_context_item(item: Any) -> Dict[str, Any]:
    """
    rag_core.generate_answer가 반환하는 컨텍스트를
    공통 포맷으로 맞추는 함수.

    기대 형태:
        - (text, score, keyword)
        - dict: {"text": ..., "score": ..., "keyword": ...}
    """
    if isinstance(item, dict):
        return {
            "text": item.get("text", ""),
            "score": float(item.get("score", 0.0)) if "score" in item else None,
            "keyword": item.get("keyword", ""),
        }

    if isinstance(item, tuple):
        # (text, score) or (text, score, keyword)
        if len(item) == 2:
            text, score = item
            keyword = ""
        elif len(item) >= 3:
            text, score, keyword = item[0], item[1], item[2]
        else:
            text, score, keyword = str(item), None, ""
        return {
            "text": text,
            "score": float(score) if score is not None else None,
            "keyword": keyword,
        }

    # 그 외 이상한 형태
    return {"text": str(item), "score": None, "keyword": ""}


def print_header():
    print("\n==============================")
    print("  반도체 ALD RAG 챗봇 (CLI)")
    print("==============================")

    if MODEL_INFO:
        dev = MODEL_INFO.get("device", "unknown")
        embed = MODEL_INFO.get("embed_model", "unknown")
        llm = MODEL_INFO.get("llm_model", "unknown")
        n_docs = MODEL_INFO.get("num_docs", "unknown")
        keywords = MODEL_INFO.get("keywords", [])
        print(f"[Device] {dev}")
        print(f"[Embed ] {embed}")
        print(f"[LLM   ] {llm}")
        print(f"[Docs  ] {n_docs} 개 로드됨")
        if keywords:
            print(f"[KWs   ] {', '.join(keywords)}")
    else:
        print("[INFO] MODEL_INFO 없음 (rag_core 최신 버전 아니어도 동작은 함)")

    print("==============================")


# ==============================
# 2) 모드별 실행
# ==============================

def run_stats_mode():
    if get_keyword_stats is None:
        print("[stats] rag_core.get_keyword_stats 가 정의돼 있지 않음.")
        return

    stats = get_keyword_stats()
    print("\n[키워드 통계]")
    if not stats:
        print("- 등록된 문서가 없거나 키워드 정보가 없음.")
        return

    for kw in sorted(stats.keys()):
        print(f"- {kw}: {stats[kw]} 문장")


def call_generate(
    question: str,
    args: argparse.Namespace,
):
    """
    generate_answer를 공통으로 호출하는 래퍼
    (filter_keyword / context_only / debug 반영)
    """
    filter_kw = args.filter_keyword.strip() or None

    answer, retrieved = generate_answer(
        question,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        filter_keyword=filter_kw,
        context_only=args.context_only,
        debug=args.debug,
    )
    return answer, retrieved


def run_once_mode(args: argparse.Namespace, log_path: Path):
    q = args.question.strip()
    if not q:
        q = input("질문 입력: ").strip()
        if not q:
            print("질문이 비어있어서 종료.")
            return

    print("\n[검색 + 답변 생성 중...]\n")

    answer, retrieved = call_generate(q, args)

    # 콘솔 출력
    print("[질문]")
    print(q)

    print("\n[모델 답변]")
    print(answer)

    print("\n[참고한 컨텍스트]")
    if not retrieved:
        print("(검색된 컨텍스트 없음)")
    else:
        for idx, item in enumerate(retrieved, start=1):
            ctx = format_context_item(item)
            score_str = f"{ctx['score']:.3f}" if ctx["score"] is not None else "N/A"
            kw_str = f" ({ctx['keyword']})" if ctx["keyword"] else ""
            print(f"{idx:02d}. score={score_str}{kw_str}")
            print("    " + ctx["text"])

    # 로그 기록
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_write(log_path, f"[{ts}] Q: {q}")
    log_write(log_path, f"[{ts}] A: {answer}")
    log_write(log_path, f"[{ts}] CONTEXTS:")
    for idx, item in enumerate(retrieved or [], start=1):
        ctx = format_context_item(item)
        log_write(
            log_path,
            f"  {idx:02d}. score={ctx['score']} kw={ctx['keyword']} text={ctx['text']}",
        )
    log_write(log_path, "-" * 60)


def run_chat_mode(args: argparse.Namespace, log_path: Path):
    print_header()
    print("엔터만 치면 종료.\n")

    while True:
        q = input("\n질문 입력: ").strip()
        if not q:
            print("끝냄!")
            break

        print("\n[검색 + 답변 생성 중...]\n")

        answer, retrieved = call_generate(q, args)

        print("[모델 답변]")
        print(answer)

        print("\n[참고한 컨텍스트]")
        if not retrieved:
            print("(검색된 컨텍스트 없음)")
        else:
            for idx, item in enumerate(retrieved, start=1):
                ctx = format_context_item(item)
                score_str = f"{ctx['score']:.3f}" if ctx["score"] is not None else "N/A"
                kw_str = f" ({ctx['keyword']})" if ctx["keyword"] else ""
                print(f"{idx:02d}. score={score_str}{kw_str}")
                print("    " + ctx["text"])

        # 로그 기록
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_write(log_path, f"[{ts}] Q: {q}")
        log_write(log_path, f"[{ts}] A: {answer}")
        log_write(log_path, f"[{ts}] CONTEXTS:")
        for idx, item in enumerate(retrieved or [], start=1):
            ctx = format_context_item(item)
            log_write(
                log_path,
                f"  {idx:02d}. score={ctx['score']} kw={ctx['keyword']} text={ctx['text']}",
            )
        log_write(log_path, "-" * 60)


# ==============================
# 3) 메인
# ==============================

if __name__ == "__main__":
    args = parse_args()
    log_path = init_log_file(args.log_file)

    if args.mode == "stats":
        run_stats_mode()
    elif args.mode == "once":
        run_once_mode(args, log_path)
    else:  # chat
        run_chat_mode(args, log_path)
