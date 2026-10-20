#!/usr/bin/env python3
# 개선된 학습 데이터 생성 스크립트: docs_ald.json → 더 다양하고 자연스러운 Q&A 쌍 생성

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# 프로젝트 루트 경로 추가
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# LLM 기반 답변 생성을 위한 임포트 (선택적)
LLM_AVAILABLE = False
TOKENIZER = None
LLM = None
DEVICE = None

try:
    from rag_core import _init_models_if_needed
    import torch  # type: ignore
    LLM_AVAILABLE = True
except Exception as e:
    print(f"[!] LLM을 사용한 답변 생성은 비활성화됩니다: {e}")

DOCS_PATH = BASE_DIR / "docs" / "docs_ald.json"
FEEDBACK_PATH = BASE_DIR / "feedback" / "feedback_data.json"
OUTPUT_DIR = BASE_DIR / "finetuning" / "data"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
EVAL_FILE = OUTPUT_DIR / "eval.jsonl"


def load_docs() -> List[Dict[str, Any]]:
    """docs_ald.json 로드"""
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"문서 파일을 찾을 수 없습니다: {DOCS_PATH}")
    
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "documents" in data:
        return data["documents"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"잘못된 문서 형식: {DOCS_PATH}")


def load_feedback() -> List[Dict[str, Any]]:
    """피드백 데이터 로드 (실제 사용자 질문 포함)"""
    if not FEEDBACK_PATH.exists():
        return []
    
    try:
        with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("feedbacks", [])
    except:
        return []


def generate_llm_answer(question: str, contexts: List[tuple], use_llm: bool = False) -> Optional[str]:
    """LLM을 사용하여 더 자연스러운 답변 생성"""
    if not use_llm or not LLM_AVAILABLE:
        return None
    
    try:
        # rag_core에서 모델 가져오기
        from rag_core import TOKENIZER as RAG_TOKENIZER, LLM as RAG_LLM, DEVICE as RAG_DEVICE
        
        if RAG_LLM is None or RAG_TOKENIZER is None:
            return None
        
        # 컨텍스트 형식: (text, score, keyword, doc_id)
        ctx_lines = []
        for ctx in contexts:
            if len(ctx) >= 3:
                text, _, kw = ctx[0], ctx[1], ctx[2]
                ctx_lines.append(f"- ({kw}) {text}")
        
        if not ctx_lines:
            return None
        
        ctx = "\n".join(ctx_lines)
        
        # RAG와 동일한 프롬프트 형식
        system_prompt = (
            "반도체 ALD 전문가로서 답변하세요.\n"
            "문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n"
            "반드시 한국어로만 답변해야 합니다."
        )
        
        user_prompt = f"""아래는 관련 문맥이야:

{ctx}

[질문]
{question}

위 문맥만 근거로 한국어로 답변해줘."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        full_prompt = RAG_TOKENIZER.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        inputs = RAG_TOKENIZER(full_prompt, return_tensors="pt")
        
        # 디바이스 처리
        if hasattr(RAG_LLM, "device"):
            model_device = RAG_LLM.device
        elif hasattr(RAG_LLM, "hf_device_map"):
            first_param = next(RAG_LLM.parameters())
            model_device = first_param.device
        else:
            model_device = RAG_DEVICE if RAG_DEVICE is not None else torch.device("cpu")
        
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = RAG_LLM.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=RAG_TOKENIZER.eos_token_id
            )
        
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = RAG_TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()
        
        # 최소 길이 체크
        if len(answer) < 20:
            return None
        
        return answer
    except Exception as e:
        print(f"[!] LLM 답변 생성 실패: {e}")
        return None


def generate_natural_questions(text: str, keywords: List[str]) -> List[str]:
    """텍스트와 키워드로부터 자연스러운 질문 생성"""
    questions = []
    
    if not keywords:
        return questions
    
    # 다양한 질문 패턴 (자연스러운 표현 포함)
    question_templates = [
        # 기본 질문 (30%)
        (f"{keywords[0]}는 무엇인가요?", 0.3),
        (f"{keywords[0]}의 역할은 무엇인가요?", 0.2),
        (f"{keywords[0]}에 대해 설명해주세요.", 0.2),
        
        # 자연스러운 질문 (40%)
        (f"{keywords[0]}가 뭐야?", 0.1),
        (f"{keywords[0]}는 뭐하는 거야?", 0.1),
        (f"{keywords[0]}에 대해 알려줘.", 0.1),
        (f"{keywords[0]}가 왜 중요한가요?", 0.1),
        (f"{keywords[0]}는 어떻게 작동하나요?", 0.1),
        (f"{keywords[0]}의 원리는?", 0.1),
        
        # 상황 기반 질문 (20%)
        (f"{keywords[0]} 문제가 생기면 어떻게 되나요?", 0.05),
        (f"{keywords[0]}를 제어할 때 주의할 점은?", 0.05),
        (f"{keywords[0]} 설정이 잘못되면?", 0.05),
        (f"{keywords[0]}가 불안정하면 어떤 영향이 있나요?", 0.05),
    ]
    
    # 키워드가 2개 이상인 경우
    if len(keywords) >= 2:
        question_templates.extend([
            (f"{keywords[0]}와 {keywords[1]}의 차이는?", 0.05),
            (f"{keywords[0]}와 {keywords[1]}는 어떤 관계인가요?", 0.05),
            (f"{keywords[0]}와 {keywords[1]} 중 어느 게 더 중요해요?", 0.03),
            (f"{keywords[0]}가 {keywords[1]}에 미치는 영향은?", 0.03),
        ])
    
    # 키워드가 3개 이상인 경우
    if len(keywords) >= 3:
        question_templates.extend([
            (f"{keywords[0]}, {keywords[1]}, {keywords[2]}의 관계는?", 0.02),
            (f"{keywords[0]}와 {keywords[1]}가 {keywords[2]}에 주는 영향은?", 0.02),
        ])
    
    # 장비 구조 관련 키워드에 대한 특별 질문
    equipment_keywords = ["VG12", "VG13", "로드락", "Load Lock", "Valve", "챔버", "Chamber"]
    if any(kw in keywords[0] for kw in equipment_keywords):
        question_templates.extend([
            (f"{keywords[0]}는 어디에 위치하나요?", 0.1),
            (f"{keywords[0]}는 어떤 장비인가요?", 0.1),
            (f"{keywords[0]}의 구조는?", 0.05),
            (f"{keywords[0]}는 어떻게 연결되어 있나요?", 0.05),
        ])
    
    # VG12/VG13 역할 구분 질문
    if "VG12" in keywords[0] or "VG13" in keywords[0]:
        question_templates.extend([
            (f"{keywords[0]}는 센서인가요 밸브인가요?", 0.15),
            (f"{keywords[0]}의 정확한 역할은?", 0.1),
            (f"{keywords[0]}는 무엇을 측정하나요?", 0.1),
        ])
    
    # 텍스트 내용 기반 질문 생성
    text_lower = text.lower()
    if "압력" in text or "pressure" in text_lower:
        question_templates.append((f"{keywords[0]}에서 압력이 중요한 이유는?", 0.05))
    if "온도" in text or "temperature" in text_lower:
        question_templates.append((f"{keywords[0]}에서 온도는 어떤 영향을 주나요?", 0.05))
    if "유량" in text or "flow" in text_lower:
        question_templates.append((f"{keywords[0]}에서 유량 제어가 중요한가요?", 0.05))
    if "불안정" in text or "오류" in text or "문제" in text:
        question_templates.append((f"{keywords[0]}에서 문제가 생기면 어떻게 되나요?", 0.05))
    if "센서" in text or "sensor" in text_lower:
        question_templates.append((f"{keywords[0]}는 센서인가요?", 0.1))
    if "밸브" in text or "valve" in text_lower:
        question_templates.append((f"{keywords[0]}는 밸브인가요?", 0.1))
    
    # 가중치 기반 샘플링 (더 다양한 질문 생성)
    selected = []
    for template, weight in question_templates:
        if random.random() < weight * 2:  # 가중치를 2배로 해서 더 많이 선택
            selected.append(template)
    
    # 최소 2개, 최대 8개 질문
    if len(selected) < 2:
        selected = [t[0] for t in question_templates[:2]]
    elif len(selected) > 8:
        selected = random.sample(selected, 8)
    
    return selected


def create_enhanced_qa_pairs(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """문서 리스트에서 개선된 질문-답변 쌍 생성"""
    qa_pairs = []
    
    # 키워드별 문서 그룹화 (더 긴 답변 생성용)
    keyword_to_docs = {}
    for doc in docs:
        keywords = doc.get("keywords", [])
        text = str(doc.get("text", "")).strip()
        
        if not text or not keywords:
            continue
        
        for keyword in keywords:
            if keyword not in keyword_to_docs:
                keyword_to_docs[keyword] = []
            keyword_to_docs[keyword].append(text)
    
    for doc in docs:
        text = str(doc.get("text", "")).strip()
        keywords = doc.get("keywords", [])
        
        if not text or not keywords:
            continue
        
        # 자연스러운 질문 생성
        questions = generate_natural_questions(text, keywords)
        
        for question in questions:
            # RAG 스타일 프롬프트 형식 (messages 기반)
            # 컨텍스트 형식: (키워드) 텍스트
            context_text = f"- ({keywords[0]}) {text}"
            contexts = [(text, 0.8, keywords[0], None)]
            
            # 30% 확률로 LLM 기반 답변 생성
            answer = text  # 기본값: 원본 텍스트
            if LLM_AVAILABLE and random.random() < 0.3:
                llm_answer = generate_llm_answer(question, contexts, use_llm=True)
                if llm_answer:
                    answer = llm_answer
            
            qa_pairs.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "반도체 ALD 전문가로서 답변하세요.\n문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n반드시 한국어로만 답변해야 합니다."
                    },
                    {
                        "role": "user",
                        "content": f"""아래는 관련 문맥이야:

{context_text}

[질문]
{question}

위 문맥만 근거로 한국어로 답변해줘."""
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            })
        
        # 키워드별로 관련 문서들을 결합한 더 긴 답변 생성 (30% 확률)
        if random.random() < 0.3 and len(keywords) > 0:
            main_keyword = keywords[0]
            related_texts = keyword_to_docs.get(main_keyword, [])
            
            if len(related_texts) > 1:
                # 관련 문서 2-3개 결합
                combined_texts = random.sample(related_texts, min(3, len(related_texts)))
                combined_answer = " ".join(combined_texts)
                
                # 더 구체적인 질문
                detailed_questions = [
                    f"{main_keyword}에 대해 자세히 설명해주세요.",
                    f"{main_keyword}의 모든 측면을 알려주세요.",
                    f"{main_keyword}와 관련된 모든 정보를 주세요.",
                ]
                
                for question in detailed_questions:
                    # 여러 컨텍스트 결합
                    context_lines = [f"- ({main_keyword}) {txt}" for txt in combined_texts]
                    context_text = "\n".join(context_lines)
                    contexts = [(txt, 0.8, main_keyword, None) for txt in combined_texts]
                    
                    # 30% 확률로 LLM 기반 답변 생성
                    answer = combined_answer  # 기본값: 결합된 텍스트
                    if LLM_AVAILABLE and random.random() < 0.3:
                        llm_answer = generate_llm_answer(question, contexts, use_llm=True)
                        if llm_answer:
                            answer = llm_answer
                    
                    qa_pairs.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": "반도체 ALD 전문가로서 답변하세요.\n문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n반드시 한국어로만 답변해야 합니다."
                            },
                            {
                                "role": "user",
                                "content": f"""아래는 관련 문맥이야:

{context_text}

[질문]
{question}

위 문맥만 근거로 한국어로 답변해줘."""
                            },
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        ]
                    })
    
    return qa_pairs


def create_keyword_combination_qa_improved(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """개선된 키워드 조합 기반 질문-답변 쌍 생성"""
    qa_pairs = []
    
    # 키워드별 문서 그룹화
    keyword_to_docs = {}
    for doc in docs:
        keywords = doc.get("keywords", [])
        text = str(doc.get("text", "")).strip()
        
        if not text or not keywords:
            continue
        
        for keyword in keywords:
            if keyword not in keyword_to_docs:
                keyword_to_docs[keyword] = []
            keyword_to_docs[keyword].append(text)
    
    # 2개 키워드 조합
    keywords_list = list(keyword_to_docs.keys())
    for i, kw1 in enumerate(keywords_list):
        for kw2 in keywords_list[i+1:]:
            # 두 키워드가 모두 포함된 문서 찾기
            combined_texts = []
            for doc in docs:
                doc_keywords = doc.get("keywords", [])
                if kw1 in doc_keywords and kw2 in doc_keywords:
                    combined_texts.append(str(doc.get("text", "")).strip())
            
            if combined_texts:
                answer = " ".join(combined_texts[:3])  # 최대 3개 결합
                
                # 다양한 질문 패턴
                combo_questions = [
                    f"{kw1}와 {kw2}의 관계는 무엇인가요?",
                    f"{kw1}와 {kw2}는 어떻게 연관되어 있나요?",
                    f"{kw1}가 {kw2}에 주는 영향은?",
                    f"{kw1}와 {kw2} 중 어느 게 더 중요해요?",
                    f"{kw1}와 {kw2}를 함께 사용할 때 주의할 점은?",
                ]
                
                for question in combo_questions[:2]:  # 각 조합당 2개 질문
                    # 키워드 조합 컨텍스트
                    context_lines = [f"- ({kw1}) {txt}" if kw1 in txt else f"- ({kw2}) {txt}" for txt in combined_texts[:3]]
                    context_text = "\n".join(context_lines)
                    
                    qa_pairs.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": "반도체 ALD 전문가로서 답변하세요.\n문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n반드시 한국어로만 답변해야 합니다."
                            },
                            {
                                "role": "user",
                                "content": f"""아래는 관련 문맥이야:

{context_text}

[질문]
{question}

위 문맥만 근거로 한국어로 답변해줘."""
                            },
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        ]
                    })
    
    return qa_pairs


def create_equipment_structure_qa(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """장비 구조 및 연결 관계 기반 질문-답변 쌍 생성"""
    qa_pairs = []
    
    # 장비 구조 관련 키워드
    equipment_keywords = {
        "VG12": ["VG12", "센서", "압력 측정", "진공 게이지", "메인 챔버"],
        "VG13": ["VG13", "센서", "압력 측정", "로드락", "인터미디어트"],
        "로드락": ["로드락", "Load Lock", "챔버", "웨이퍼 이송", "진공 전환"],
        "Valve": ["Valve", "밸브", "APC", "개도율", "제어"],
        "챔버": ["챔버", "Chamber", "메인 챔버", "공정 챔버"]
    }
    
    # 키워드별 문서 그룹화
    keyword_to_docs = {}
    for doc in docs:
        keywords = doc.get("keywords", [])
        text = str(doc.get("text", "")).strip()
        
        if not text or not keywords:
            continue
        
        for keyword in keywords:
            if keyword not in keyword_to_docs:
                keyword_to_docs[keyword] = []
            keyword_to_docs[keyword].append(text)
    
    # 1. VG12/VG13 역할 구분 질문 (센서 vs 밸브)
    vg12_docs = keyword_to_docs.get("VG12", [])
    vg13_docs = keyword_to_docs.get("VG13", [])
    
    if vg12_docs:
        # VG12 역할 명확화
        vg12_texts = " ".join(vg12_docs[:3])
        vg12_questions = [
            "VG12는 무엇인가요?",
            "VG12의 역할은 무엇인가요?",
            "VG12는 센서인가요 밸브인가요?",
            "VG12는 어떤 장비인가요?",
            "VG12는 어디에 위치하나요?",
        ]
        for q in vg12_questions:
            context_text = "\n".join([f"- (VG12) {txt}" for txt in vg12_docs[:3]])
            qa_pairs.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "반도체 ALD 전문가로서 답변하세요.\n문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n반드시 한국어로만 답변해야 합니다."
                    },
                    {
                        "role": "user",
                        "content": f"""아래는 관련 문맥이야:

{context_text}

[질문]
{q}

위 문맥만 근거로 한국어로 답변해줘."""
                    },
                    {
                        "role": "assistant",
                        "content": vg12_texts
                    }
                ]
            })
    
    if vg13_docs:
        # VG13 역할 명확화
        vg13_texts = " ".join(vg13_docs[:3])
        vg13_questions = [
            "VG13는 무엇인가요?",
            "VG13의 역할은 무엇인가요?",
            "VG13는 센서인가요 밸브인가요?",
            "VG13는 어떤 장비인가요?",
            "VG13는 어디에 위치하나요?",
        ]
        for q in vg13_questions:
            context_text = "\n".join([f"- (VG13) {txt}" for txt in vg13_docs[:3]])
            qa_pairs.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "반도체 ALD 전문가로서 답변하세요.\n문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n반드시 한국어로만 답변해야 합니다."
                    },
                    {
                        "role": "user",
                        "content": f"""아래는 관련 문맥이야:

{context_text}

[질문]
{q}

위 문맥만 근거로 한국어로 답변해줘."""
                    },
                    {
                        "role": "assistant",
                        "content": vg13_texts
                    }
                ]
            })
    
    # 2. 장비 구조 및 연결 관계 질문
    structure_combinations = [
        ("VG12", "VG13", "압력 차이", "누설 여부", "진공 안정성"),
        ("VG12", "로드락", "압력 매칭", "웨이퍼 인입", "오염 방지"),
        ("VG13", "로드락", "압력 제어", "웨이퍼 이송", "진공 전환"),
        ("로드락", "챔버", "압력 차이", "웨이퍼 이동", "오염"),
        ("VG12", "Valve", "압력 제어", "피드백", "동기화"),
        ("VG13", "Valve", "압력 제어", "피드백", "배기 효율"),
    ]
    
    for kw1, kw2, relation1, relation2, relation3 in structure_combinations:
        # 두 키워드가 모두 포함된 문서 찾기
        combined_texts = []
        for doc in docs:
            doc_keywords = doc.get("keywords", [])
            if kw1 in doc_keywords and kw2 in doc_keywords:
                combined_texts.append(str(doc.get("text", "")).strip())
        
        if combined_texts:
            answer = " ".join(combined_texts[:3])
            
            # 장비 구조 기반 질문
            structure_questions = [
                f"{kw1}와 {kw2}의 연결 관계는 무엇인가요?",
                f"{kw1}와 {kw2}는 어떻게 연결되어 있나요?",
                f"{kw1}와 {kw2}의 구조적 관계는?",
                f"{kw1}와 {kw2} 사이의 {relation1}는?",
                f"{kw1}와 {kw2}의 {relation2} 관계는?",
                f"{kw1}와 {kw2}가 {relation3}에 미치는 영향은?",
            ]
            
            for q in structure_questions[:3]:  # 각 조합당 3개 질문
                context_lines = [f"- ({kw1}) {txt}" if kw1 in txt else f"- ({kw2}) {txt}" for txt in combined_texts[:3]]
                context_text = "\n".join(context_lines)
                
                qa_pairs.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "반도체 ALD 전문가로서 답변하세요.\n문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n반드시 한국어로만 답변해야 합니다."
                        },
                        {
                            "role": "user",
                            "content": f"""아래는 관련 문맥이야:

{context_text}

[질문]
{q}

위 문맥만 근거로 한국어로 답변해줘."""
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ]
                })
    
    # 3. 3개 이상 장비 구조 질문
    triple_combinations = [
        ("로드락", "VG12", "VG13", "압력 차이", "누설 판단"),
        ("로드락", "Valve", "VG13", "압력 제어", "이송 안정성"),
        ("챔버", "VG12", "Valve", "압력 제어", "공정 안정성"),
    ]
    
    for kw1, kw2, kw3, relation1, relation2 in triple_combinations:
        combined_texts = []
        for doc in docs:
            doc_keywords = doc.get("keywords", [])
            if kw1 in doc_keywords and kw2 in doc_keywords and kw3 in doc_keywords:
                combined_texts.append(str(doc.get("text", "")).strip())
        
        if combined_texts:
            answer = " ".join(combined_texts[:3])
            triple_questions = [
                f"{kw1}, {kw2}, {kw3}의 구조적 관계는?",
                f"{kw1}와 {kw2}와 {kw3}는 어떻게 연결되어 있나요?",
                f"{kw1}, {kw2}, {kw3}의 {relation1}는?",
                f"{kw1}, {kw2}, {kw3}가 {relation2}에 미치는 영향은?",
            ]
            
            for q in triple_questions[:2]:
                context_lines = [f"- ({kw1}) {txt}" for txt in combined_texts[:3]]
                context_text = "\n".join(context_lines)
                
                qa_pairs.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "반도체 ALD 전문가로서 답변하세요.\n문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n반드시 한국어로만 답변해야 합니다."
                        },
                        {
                            "role": "user",
                            "content": f"""아래는 관련 문맥이야:

{context_text}

[질문]
{q}

위 문맥만 근거로 한국어로 답변해줘."""
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ]
                })
    
    return qa_pairs


def create_feedback_based_qa(feedbacks: List[Dict[str, Any]], docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """피드백 데이터 기반 Q&A 생성"""
    qa_pairs = []
    
    if not feedbacks:
        return qa_pairs
    
    # 피드백에서 좋은 질문-답변 쌍 추출
    for fb in feedbacks:
        question = fb.get("question", "").strip()
        answer = fb.get("answer", "").strip()
        
        if question and answer and len(answer) > 20:  # 너무 짧은 답변 제외
            # 피드백에서 컨텍스트 추출
            contexts = fb.get("contexts", [])
            if contexts:
                context_lines = [f"- ({ctx.get('keyword', '')}) {ctx.get('text', '')}" for ctx in contexts]
                context_text = "\n".join(context_lines)
            else:
                context_text = "- (일반) 관련 정보"
            
            qa_pairs.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "반도체 ALD 전문가로서 답변하세요.\n문서에 없는 내용은 '관련 정보가 부족합니다'라고 명시하세요.\n반드시 한국어로만 답변해야 합니다."
                    },
                    {
                        "role": "user",
                        "content": f"""아래는 관련 문맥이야:

{context_text}

[질문]
{question}

위 문맥만 근거로 한국어로 답변해줘."""
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            })
    
    return qa_pairs


def save_jsonl(data: List[Dict[str, str]], filepath: Path):
    """JSONL 형식으로 저장"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with filepath.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[+] 저장 완료: {filepath} ({len(data)}개 항목)")


def main():
    print("[+] 개선된 학습 데이터 생성 시작...")
    
    # LLM 초기화 (사용 가능한 경우)
    global LLM_AVAILABLE
    if LLM_AVAILABLE:
        try:
            print("[+] LLM 초기화 중... (답변 생성에 사용)")
            _init_models_if_needed()
            from rag_core import TOKENIZER as RAG_TOKENIZER, LLM as RAG_LLM
            if RAG_LLM is not None and RAG_TOKENIZER is not None:
                print("[+] LLM 기반 답변 생성 활성화 (30% 확률)")
            else:
                LLM_AVAILABLE = False
                print("[!] LLM 로드 실패, 단순 텍스트 결합 방식 사용")
        except Exception as e:
            LLM_AVAILABLE = False
            print(f"[!] LLM 초기화 실패: {e}, 단순 텍스트 결합 방식 사용")
    else:
        print("[!] LLM 사용 불가, 단순 텍스트 결합 방식 사용")
    
    # 문서 로드
    docs = load_docs()
    print(f"[+] 문서 로드 완료: {len(docs)}개")
    
    # 피드백 로드
    feedbacks = load_feedback()
    print(f"[+] 피드백 로드 완료: {len(feedbacks)}개")
    
    # Q&A 쌍 생성
    qa_pairs = []
    
    # 1. 개선된 기본 Q&A 쌍
    print("[+] 개선된 기본 Q&A 쌍 생성 중...")
    basic_qa = create_enhanced_qa_pairs(docs)
    qa_pairs.extend(basic_qa)
    print(f"  - 생성된 기본 Q&A: {len(basic_qa)}개")
    
    # 2. 개선된 키워드 조합 Q&A
    print("[+] 개선된 키워드 조합 Q&A 생성 중...")
    combo_qa = create_keyword_combination_qa_improved(docs)
    qa_pairs.extend(combo_qa)
    print(f"  - 생성된 조합 Q&A: {len(combo_qa)}개")
    
    # 3. 장비 구조 기반 Q&A (피드백 반영)
    print("[+] 장비 구조 기반 Q&A 생성 중...")
    structure_qa = create_equipment_structure_qa(docs)
    qa_pairs.extend(structure_qa)
    print(f"  - 생성된 장비 구조 Q&A: {len(structure_qa)}개")
    
    # 4. 피드백 기반 Q&A
    if feedbacks:
        print("[+] 피드백 기반 Q&A 생성 중...")
        feedback_qa = create_feedback_based_qa(feedbacks, docs)
        qa_pairs.extend(feedback_qa)
        print(f"  - 생성된 피드백 Q&A: {len(feedback_qa)}개")
    
    # 중복 제거
    seen = set()
    unique_qa = []
    for qa in qa_pairs:
        # messages 형식인지 확인
        if "messages" in qa:
            # messages 형식: 질문과 답변 추출
            if len(qa["messages"]) >= 3:
                user_content = qa["messages"][1].get("content", "")
                assistant_content = qa["messages"][2].get("content", "")
                # 질문 부분만 추출 (문맥 제외)
                if "[질문]" in user_content:
                    question = user_content.split("[질문]")[-1].strip().split("\n")[0]
                else:
                    question = user_content
                key = (question, assistant_content)
            else:
                continue
        else:
            # 기존 형식 (input/output)
            key = (qa.get("input", ""), qa.get("output", ""))
        
        if key not in seen:
            seen.add(key)
            unique_qa.append(qa)
    
    print(f"[+] 총 Q&A 쌍: {len(unique_qa)}개 (중복 제거 후)")
    
    # 질문 패턴 분석
    question_patterns = {}
    for qa in unique_qa[:100]:  # 샘플 100개
        # messages 형식인지 확인
        if "messages" in qa:
            q = qa["messages"][1]["content"] if len(qa["messages"]) > 1 else ""
            # 질문 부분만 추출
            if "[질문]" in q:
                q = q.split("[질문]")[-1].strip().split("\n")[0]
        else:
            q = qa.get("input", "")
        if "는 무엇인가요?" in q or "은 무엇인가요?" in q:
            pattern = "X는 무엇인가요?"
        elif "의 역할은" in q:
            pattern = "X의 역할은?"
        elif "에 대해 설명" in q:
            pattern = "X에 대해 설명"
        elif "뭐야" in q or "뭐하는" in q:
            pattern = "자연스러운 질문"
        elif "관계" in q or "차이" in q:
            pattern = "비교/관계 질문"
        else:
            pattern = "기타"
        
        question_patterns[pattern] = question_patterns.get(pattern, 0) + 1
    
    print(f"[+] 질문 패턴 분포 (샘플 100개):")
    for pattern, count in sorted(question_patterns.items(), key=lambda x: -x[1]):
        print(f"  - {pattern}: {count}개")
    
    # 학습/검증 분할 (8:2)
    random.seed(42)
    random.shuffle(unique_qa)
    
    split_idx = int(len(unique_qa) * 0.8)
    train_data = unique_qa[:split_idx]
    eval_data = unique_qa[split_idx:]
    
    # 저장
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(eval_data, EVAL_FILE)
    
    print(f"[+] 완료!")
    print(f"  - 학습 데이터: {len(train_data)}개")
    print(f"  - 검증 데이터: {len(eval_data)}개")
    
    # 평균 답변 길이 계산 (messages 형식 지원)
    answer_lengths = []
    for qa in unique_qa:
        if "messages" in qa:
            if len(qa["messages"]) > 2:
                answer_lengths.append(len(qa["messages"][2]["content"]))
        else:
            answer_lengths.append(len(qa.get("output", "")))
    
    if answer_lengths:
        print(f"  - 평균 답변 길이: {sum(answer_lengths) / len(answer_lengths):.1f}자")


if __name__ == "__main__":
    main()

