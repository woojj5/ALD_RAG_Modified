#!/usr/bin/env python3
"""
docs_ald.json에 유의미한 데이터를 생성/추가하는 도구

1. LLM 기반 생성: 기존 키워드와 텍스트를 기반으로 유사한 내용 생성
2. 웹 검색 기반: 웹에서 반도체/ALD 관련 정보 검색 및 추출
3. 템플릿 기반: 미리 정의된 템플릿으로 데이터 생성

사용 예시:
    python generate_docs.py llm --keyword ALD --count 5
    python generate_docs.py web --keyword "plasma ALD" --count 3
    python generate_docs.py template --keyword MFC
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# rag_core 모듈 경로 추가
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ~/ald-rag-lab
sys.path.insert(0, str(BASE_DIR))

DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"

from rag_core import generate_text


# ==============================
# 1) 문서 로딩/저장
# ==============================

def load_docs() -> List[Dict[str, Any]]:
    """Load the canonical list-based knowledge-base format."""
    if not DOCS_PATH.exists():
        return []
    
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = data.get("documents", [])
    if not isinstance(documents, list) or not all(isinstance(item, dict) for item in documents):
        raise ValueError("docs_ald.json documents must be a list of objects")
    return documents


def save_docs(documents: List[Dict[str, Any]]) -> None:
    """Save documents without changing the knowledge-base schema."""
    data = {"documents": documents}
    with DOCS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[+] 저장 완료: {len(documents)}개 문서")


# ==============================
# 2) LLM 기반 데이터 생성
# ==============================

def generate_with_llm(keyword: str, existing_texts: List[str], count: int = 3) -> List[str]:
    """LLM을 사용하여 키워드 관련 텍스트 생성"""
    # 기존 텍스트를 참고하여 유사한 스타일로 생성
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
    
    # 생성된 텍스트를 문장별로 분리
    lines = [line.strip() for line in answer.strip().split("\n") if line.strip()]
    
    # 번호나 불필요한 접두사 제거
    cleaned_lines = []
    for line in lines:
        # "1. ", "- ", "* " 같은 접두사 제거
        for prefix in ["1.", "2.", "3.", "4.", "5.", "- ", "* ", "• "]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if line and len(line) > 10:  # 너무 짧은 것 제외
            cleaned_lines.append(line)
    
    return cleaned_lines[:count]


# ==============================
# 3) 웹 검색 기반 데이터 생성
# ==============================

def search_web(keyword: str, query: str, count: int = 3) -> List[str]:
    """웹 검색을 통해 관련 정보 수집 (간단한 예시)"""
    print(f"[!] 웹 검색 기능은 구현 예정입니다.")
    print(f"    키워드: {keyword}, 검색어: {query}")
    print(f"    실제 구현 시 requests, BeautifulSoup 등을 사용할 수 있습니다.")
    
    # TODO: 실제 웹 검색 구현
    # - Google Custom Search API
    # - Wikipedia API
    # - arXiv API (학술 논문)
    
    return []


# ==============================
# 4) 템플릿 기반 데이터 생성
# ==============================

TEMPLATES: Dict[str, List[str]] = {
    "ALD": [
        "{keyword} 공정에서 {parameter}는 {value} 범위에서 설정된다.",
        "{keyword}의 {aspect}는 {factor}에 의해 결정된다.",
        "{keyword} 공정 중 {step} 단계에서 {condition}을 확인해야 한다.",
    ],
    "Precursor": [
        "{keyword} {type}는 {temperature}℃에서 {state} 상태로 주입된다.",
        "{keyword}의 {property}는 {value}로 측정된다.",
        "{keyword} {name}는 {reaction} 반응을 일으킨다.",
    ],
    "MFC": [
        "{keyword}는 {range} sccm 범위에서 {precision}% 정밀도로 유량을 제어한다.",
        "{keyword} {model}의 응답 시간은 {time}초 이내이다.",
        "{keyword} 이상 시 {symptom} 현상이 발생할 수 있다.",
    ],
    "Flow": [
        "{keyword}는 {unit} 단위로 측정되며, {range} 범위가 일반적이다.",
        "{keyword} {type}는 {condition} 조건에서 {value}로 설정된다.",
        "{keyword} 이상은 {indicator}로 확인할 수 있다.",
    ],
    "Purge": [
        "{keyword} 시간은 {duration}초이며, {purpose} 목적으로 수행된다.",
        "{keyword} {gas}는 {flow} sccm로 {time}초 동안 주입된다.",
        "{keyword} 불충분 시 {problem} 문제가 발생할 수 있다.",
    ],
    "플라즈마": [
        "{keyword} {type}는 {power}W, {frequency}Hz 조건에서 생성된다.",
        "{keyword}의 {property}는 {parameter}에 따라 변화한다.",
        "{keyword} 사용 시 {benefit} 효과를 얻을 수 있다.",
    ],
    "압력": [
        "{keyword}는 {sensor}로 측정되며, {range} Torr 범위에서 유지된다.",
        "{keyword} {type}는 {valve}로 제어되어 {target} Torr로 유지된다.",
        "{keyword} 이상 시 {symptom} 현상이 나타난다.",
    ],
    "챔버": [
        "{keyword} {part}의 온도는 {temp}℃로 제어된다.",
        "{keyword} 내부 {component}는 {material}로 제작된다.",
        "{keyword} {condition} 조건에서 {process}가 진행된다.",
    ],
    "VG12": [
        "{keyword}는 {location}에 설치되어 {parameter}를 측정한다.",
        "{keyword} 측정값은 {range} Torr 범위에서 정확도를 유지한다.",
        "{keyword} 이상 시 {indicator}로 확인할 수 있다.",
    ],
    "VG13": [
        "{keyword}는 {location}의 {parameter}를 모니터링한다.",
        "{keyword} 값이 {threshold} 이상이면 {action}이 필요하다.",
        "{keyword} {type}는 {specification} 사양을 가진다.",
    ],
    "로드락": [
        "{keyword} {chamber}는 {pressure} Torr까지 진공을 형성한다.",
        "{keyword} {valve}는 {sequence} 순서로 작동한다.",
        "{keyword} {process} 중 {parameter}를 확인해야 한다.",
    ],
    "Valve": [
        "{keyword} {type}는 {actuator}로 제어되며 {response} 응답 시간을 가진다.",
        "{keyword} {state} 상태는 {sensor}로 확인할 수 있다.",
        "{keyword} 이상 시 {symptom} 패턴이 나타난다.",
    ],
    "웨이퍼": [
        "{keyword} {size}는 {diameter}mm 직경을 가진다.",
        "{keyword} {surface} 표면은 {treatment} 처리가 필요하다.",
        "{keyword} {position} 위치에서 {process}가 수행된다.",
    ],
}


def generate_from_template(keyword: str, count: int = 3) -> List[str]:
    """템플릿을 사용하여 데이터 생성"""
    templates = TEMPLATES.get(keyword, TEMPLATES.get("ALD", []))
    
    if not templates:
        print(f"[!] {keyword}에 대한 템플릿이 없습니다.")
        return []
    
    # 간단한 예시 생성 (실제로는 더 복잡한 로직 필요)
    results = []
    for i, template in enumerate(templates[:count]):
        # 템플릿의 placeholder를 실제 값으로 대체 (간단한 예시)
        text = template.format(
            keyword=keyword,
            parameter="온도",
            value="200-300",
            aspect="두께",
            factor="사이클 수",
            step="증착",
            condition="압력",
            type="A",
            temperature="150",
            state="기체",
            property="순도",
            name="TMA",
            reaction="표면 흡착",
            range="0-1000",
            precision="1",
            model="MKS",
            time="0.5",
            symptom="유량 불안정",
            unit="sccm",
            duration="10",
            purpose="잔류 가스 제거",
            gas="N2",
            flow="100",
            problem="불순물 혼입",
            power="300",
            frequency="13.56",
            benefit="낮은 온도 증착",
            sensor="피라니 게이지",
            target="0.1",
            part="하부",
            temp="200",
            component="샤워헤드",
            material="스테인리스",
            process="ALD",
            location="공정 챔버",
            threshold="1e-3",
            action="압력 확인",
            specification="0.1-1000 Torr",
            chamber="전단",
            pressure="1e-6",
            sequence="진공-가스-진공",
            size="8인치",
            diameter="200",
            surface="실리콘",
            treatment="클리닝",
            position="중앙",
            actuator="솔레노이드",
            response="100ms",
            valve_state="열림/닫힘",
            indicator="압력 변화"
        )
        results.append(text)
    
    return results


# ==============================
# 5) 메인 함수들
# ==============================

def run_llm_mode(keyword: str, count: int):
    """LLM을 사용하여 데이터 생성
    
    ⚠️ 중요: LLM이 생성한 문장은 반드시 전문가 검토가 필요합니다.
    정확성과 전문성을 보장하기 위해 실제 매뉴얼이나 문서를 참고하는 것을 강력히 권장합니다.
    """
    print(f"[+] LLM 모드: '{keyword}' 키워드에 {count}개 문장 생성")
    print(f"[!] ⚠️  주의: 생성된 문장은 반드시 전문가 검토가 필요합니다!")
    print(f"    정확한 데이터를 위해서는 extract_from_docs.py를 사용하여")
    print(f"    실제 매뉴얼이나 문서에서 추출하는 것을 권장합니다.\n")
    
    docs = load_docs()
    existing_texts = [
        str(item.get("text", ""))
        for item in docs
        if keyword in item.get("keywords", [])
    ]
    
    if not existing_texts:
        print(f"[!] 경고: '{keyword}' 키워드에 기존 데이터가 없습니다.")
        print(f"    템플릿 모드를 사용하거나 수동으로 몇 개 추가한 후 다시 시도하세요.")
        return
    
    try:
        new_texts = generate_with_llm(keyword, existing_texts, count)
        
        if not new_texts:
            print("[!] 생성된 텍스트가 없습니다.")
            return
        
        print(f"\n[생성된 문장들]")
        print("=" * 80)
        for i, text in enumerate(new_texts, 1):
            print(f"  {i}. {text}")
        print("=" * 80)
        
        print("\n[!] ⚠️  중요: 위 문장들의 정확성을 반드시 확인하세요!")
        print("    - 수치와 단위가 정확한가?")
        print("    - 기술적 내용이 올바른가?")
        print("    - 실제 공정/장비와 일치하는가?")
        
        # 추가 확인
        confirm = input(f"\n위 {len(new_texts)}개 문장을 검토 후 추가하시겠습니까? (y/n): ").strip().lower()
        if confirm == 'y':
            next_id = max(
                (item.get("id", 0) for item in docs if isinstance(item.get("id"), int)),
                default=0,
            ) + 1
            for text in new_texts:
                docs.append({"id": next_id, "keywords": [keyword], "text": text})
                next_id += 1
            save_docs(docs)
            print(f"[✓] {len(new_texts)}개 문장이 추가되었습니다.")
        else:
            print("[!] 취소되었습니다.")
            
    except Exception as e:
        print(f"[ERROR] LLM 생성 실패: {e}")
        import traceback
        traceback.print_exc()


def run_template_mode(keyword: str, count: int):
    """템플릿을 사용하여 데이터 생성"""
    print(f"[+] 템플릿 모드: '{keyword}' 키워드에 {count}개 문장 생성")
    
    new_texts = generate_from_template(keyword, count)
    
    if not new_texts:
        print(f"[!] '{keyword}'에 대한 템플릿이 없습니다.")
        return
    
    print(f"\n[생성된 문장들]")
    for i, text in enumerate(new_texts, 1):
        print(f"  {i}. {text}")
    
    confirm = input(f"\n위 {len(new_texts)}개 문장을 추가하시겠습니까? (y/n): ").strip().lower()
    if confirm == 'y':
        docs = load_docs()
        next_id = max(
            (item.get("id", 0) for item in docs if isinstance(item.get("id"), int)),
            default=0,
        ) + 1
        for text in new_texts:
            docs.append({"id": next_id, "keywords": [keyword], "text": text})
            next_id += 1
        save_docs(docs)
        print(f"[✓] {len(new_texts)}개 문장이 추가되었습니다.")


def run_web_mode(keyword: str, query: str, count: int):
    """웹 검색을 사용하여 데이터 생성"""
    print(f"[+] 웹 검색 모드: '{keyword}' 키워드, 검색어: '{query}'")
    search_web(keyword, query, count)


# ==============================
# 6) CLI
# ==============================

def parse_args():
    parser = argparse.ArgumentParser(description="docs_ald.json 데이터 생성 도구")
    sub = parser.add_subparsers(dest="mode", required=True)
    
    # LLM 모드
    p_llm = sub.add_parser("llm", help="LLM을 사용하여 데이터 생성")
    p_llm.add_argument("--keyword", type=str, required=True, help="키워드")
    p_llm.add_argument("--count", type=int, default=3, help="생성할 문장 개수 (기본: 3)")
    
    # 템플릿 모드
    p_template = sub.add_parser("template", help="템플릿을 사용하여 데이터 생성")
    p_template.add_argument("--keyword", type=str, required=True, help="키워드")
    p_template.add_argument("--count", type=int, default=3, help="생성할 문장 개수 (기본: 3)")
    
    # 웹 검색 모드
    p_web = sub.add_parser("web", help="웹 검색을 사용하여 데이터 생성 (구현 예정)")
    p_web.add_argument("--keyword", type=str, required=True, help="키워드")
    p_web.add_argument("--query", type=str, help="검색어 (없으면 키워드 사용)")
    p_web.add_argument("--count", type=int, default=3, help="생성할 문장 개수 (기본: 3)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "llm":
        run_llm_mode(args.keyword, args.count)
    elif args.mode == "template":
        run_template_mode(args.keyword, args.count)
    elif args.mode == "web":
        query = args.query or args.keyword
        run_web_mode(args.keyword, query, args.count)
