---
base_model: google/gemma-3-27b-it
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:google/gemma-3-27b-it
- lora
- transformers
---

# Gemma-ALD-LoRA

반도체 ALD 공정 전문 RAG 시스템을 위한 Gemma 3 27B 기반 LoRA Fine-tuned 모델입니다.

**참고**: 이 모델은 `qwen-ald-lora`와 동일한 구조로 구성되어 비교 가능합니다.

## Model Details

### Model Description

- **Base Model**: google/gemma-3-27b-it
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: Causal Language Modeling for RAG (Retrieval Augmented Generation)
- **Domain**: 반도체 ALD (Atomic Layer Deposition) 공정

### Model Sources

- **Base Model Repository**: https://huggingface.co/google/gemma-3-27b-it
- **Fine-tuning Dataset**: ALD 공정 관련 문서 및 피드백 데이터

## 사용 방법

### 1. Base Model 다운로드

```bash
cd finetuning/models/gemma-ald-lora
python download_base_model.py
```

**사전 준비사항:**
- Hugging Face 토큰 설정: `export HUGGINGFACE_TOKEN='your_token_here'`
- 디스크 공간: 최소 60GB 이상
- GPU: 24GB VRAM 이상 권장 (27B 모델)

### 2. Base Model 테스트

```bash
python test_base_model.py
```

### 3. Fine-tuning (LoRA Adapter 생성)

Fine-tuning 후 이 디렉토리에 다음 파일들이 생성됩니다:
- `adapter_config.json` - LoRA 설정
- `adapter_model.safetensors` - LoRA 가중치
- `tokenizer_config.json` - Tokenizer 설정
- `chat_template.jinja` - Chat template (Gemma용)
- 기타 필요한 파일들

**qwen-ald-lora와 동일한 구조:**
```
gemma-ald-lora/
├── adapter_config.json          # LoRA 설정 (qwen과 동일한 형식)
├── adapter_model.safetensors    # LoRA 가중치
├── tokenizer_config.json
├── chat_template.jinja          # Gemma용 chat template
├── special_tokens_map.json
├── tokenizer.json
└── README.md                    # 이 파일
```

### 4. RAG 시스템에서 사용

`rag_core.py`에서 모델 경로 설정:

```python
# Base model 이름
LLM_MODEL_NAME = "google/gemma-3-27b-it"

# Fine-tuned LoRA adapter 경로
FINETUNED_MODEL_PATH = BASE_DIR / "finetuning" / "models" / "gemma-ald-lora"
```

## qwen-ald-lora와의 비교

이 모델은 `qwen-ald-lora`와 동일한 구조로 구성되어 있어 직접 비교가 가능합니다:

| 항목 | qwen-ald-lora | gemma-ald-lora |
|------|---------------|----------------|
| Base Model | Qwen/Qwen2.5-7B-Instruct | google/gemma-3-27b-it |
| 모델 크기 | 7B | 27B |
| Fine-tuning | LoRA | LoRA |
| 디렉토리 구조 | 동일 | 동일 |
| 사용 방법 | 동일 | 동일 |

## 주의사항

1. **모델 크기**: 27B 모델은 7B 모델보다 훨씬 큽니다
   - 메모리 요구사항: 최소 24GB VRAM
   - 추론 속도: 더 느릴 수 있음

2. **Gemma 3 27B**: Gemma 3 27B 모델을 사용합니다

3. **Fine-tuning**: Fine-tuning 스크립트는 `finetuning/scripts/` 디렉토리에 있습니다
   - `finetune_llama.py`를 참고하여 Gemma용 fine-tuning 스크립트 작성 필요

## 문제 해결

### 다운로드 실패
- Hugging Face 토큰 확인
- 인터넷 연결 확인
- 디스크 공간 확인

### 메모리 부족
- GPU 메모리 확인: `nvidia-smi`
- 더 작은 모델 사용 고려 (gemma-3-9b-it)
- 양자화 옵션 사용

### 모델 로드 실패
- `transformers` 라이브러리 최신 버전 확인
- `peft` 라이브러리 설치 확인
- 모델 경로 확인

## Framework versions

- PEFT 0.18.0 (qwen-ald-lora와 동일)

