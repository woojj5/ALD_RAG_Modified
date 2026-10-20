# Fine-tuning 가이드

반도체 ALD 공정 전문가 모델을 Fine-tuning하는 가이드입니다.

## 📁 디렉토리 구조

```
finetuning/
├── README.md              # 이 파일
├── data/                  # 학습 데이터
│   ├── train.jsonl       # 학습용 Q&A 쌍
│   └── eval.jsonl        # 검증용 Q&A 쌍
├── models/                # Fine-tuned 모델 저장소
│   └── qwen-ald-lora/    # LoRA 어댑터 모델
└── scripts/               # 실행 스크립트
    ├── prepare_finetuning_data.py  # 데이터 생성
    ├── finetune_llama.py           # Fine-tuning 실행
    └── run_finetuning.sh           # 전체 자동 실행
```

## 🚀 빠른 시작

### 자동 실행 (권장)

```bash
cd /home/keti_spark1/ald-rag-lab
./finetuning/scripts/run_finetuning.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
1. 학습 데이터 생성
2. Fine-tuning 실행
3. 모델 저장

### 수동 실행

#### 1단계: 학습 데이터 생성

```bash
cd /home/keti_spark1/ald-rag-lab
python finetuning/scripts/prepare_finetuning_data.py
```

**생성되는 데이터:**
- `docs/docs_ald.json`에서 질문-답변 쌍 생성 (개선된 버전)
- `feedback/feedback_data.json`에서 실제 사용자 질문 포함
- 더 다양하고 자연스러운 질문 패턴 생성
- `finetuning/data/train.jsonl`, `eval.jsonl` 생성

**데이터 특징:**
- 자연스러운 질문 패턴 포함 ("뭐야?", "어떻게?")
- 평균 답변 길이: ~90자
- 키워드 조합 질문 포함
- 피드백 데이터 반영

#### 2단계: Fine-tuning 실행

**기본 실행 (학습 재개 가능):**
```bash
python finetuning/scripts/finetune_llama.py \
  --train_file finetuning/data/train.jsonl \
  --eval_file finetuning/data/eval.jsonl \
  --output_dir finetuning/models/qwen-ald-lora \
  --num_epochs 3 \
  --batch_size 4
```

**추론용 모델만 저장 (권장, 파일 크기 최소화):**
```bash
python finetuning/scripts/finetune_llama.py \
  --train_file finetuning/data/train.jsonl \
  --eval_file finetuning/data/eval.jsonl \
  --output_dir finetuning/models/qwen-ald-lora \
  --num_epochs 3 \
  --batch_size 4 \
  --save_inference_only
```

**파라미터:**
- `--model_name`: 기본 모델 (기본값: Qwen/Qwen2.5-7B-Instruct)
- `--num_epochs`: 학습 에폭 수 (기본값: 3)
- `--batch_size`: 배치 크기 (기본값: 4)
- `--learning_rate`: 학습률 (기본값: 2e-4)
- `--save_inference_only`: 추론용 모델만 저장 (optimizer 상태 제외, 체크포인트 optimizer 파일 삭제)
  - 파일 크기 대폭 감소 (optimizer.pt는 ~77MB씩)
  - 추론에는 모델 가중치만 필요하므로 이 옵션 사용 권장

#### 3단계: Fine-tuned 모델 사용

`rag_core.py`에서 Fine-tuned 모델 경로 설정:

```python
FINETUNED_MODEL_PATH = BASE_DIR / "finetuning" / "models" / "qwen-ald-lora"
```

## 📦 필요 패키지

```bash
pip install transformers datasets peft accelerate bitsandbytes
```

## ⚙️ 기술 스택

- **기본 모델**: Qwen/Qwen2.5-7B-Instruct
- **Fine-tuning 방법**: LoRA (Low-Rank Adaptation)
- **LoRA 설정**:
  - rank (r): 16
  - alpha: 32
  - target_modules: q_proj, v_proj, k_proj, o_proj
  - dropout: 0.1

## 💾 모델 저장 옵션

### 추론용 모델 저장 (`--save_inference_only`)

**추론용으로 사용할 경우 이 옵션을 사용하세요:**

- ✅ **장점**:
  - 파일 크기 대폭 감소 (optimizer.pt 파일 제거, 각각 ~77MB)
  - GitHub 업로드 시 용량 절약
  - 추론에 필요한 가중치만 저장 (adapter_model.bin 또는 adapter_model.safetensors)

- ❌ **단점**:
  - 학습 중단 후 재개 불가 (optimizer 상태 없음)
  - 체크포인트에서 optimizer 파일들이 삭제됨

**사용 예시:**
```bash
# 추론용 모델만 저장
python finetuning/scripts/finetune_llama.py \
  --save_inference_only \
  --output_dir finetuning/models/qwen-ald-lora \
  ...
```

**저장되는 파일:**
- `adapter_model.bin` 또는 `adapter_model.safetensors` (LoRA 가중치)
- `adapter_config.json` (LoRA 설정)
- `tokenizer` 관련 파일들

**저장되지 않는 파일:**
- `optimizer.pt` (optimizer 상태)
- `scheduler.pt` (scheduler 상태)
- `scaler.pt` (mixed precision scaler)
- `rng_state.pth` (랜덤 시드 상태)

## ⚠️ 주의사항

- **GPU 권장**: 16GB VRAM 이상
- **학습 시간**: 2-4시간 (데이터셋 크기 및 GPU 성능에 따라)
- **메모리**: LoRA 사용으로 메모리 효율적 (전체 모델 Fine-tuning 대비)
- **체크포인트**: 학습 중간 저장본은 `checkpoint-*/` 디렉토리에 저장됨
- **추론용 저장**: 서비스 배포 시 `--save_inference_only` 옵션 사용 권장

## 📊 데이터 통계

현재 학습 데이터:
- 학습 데이터: ~850개
- 검증 데이터: ~215개
- 평균 답변 길이: ~90자
- 질문 패턴: 다양 (자연스러운 질문 포함)

## 🔧 문제 해결

### 메모리 부족 오류
- `--batch_size`를 줄이기 (예: 2 또는 1)
- `gradient_accumulation_steps` 증가

### 학습이 너무 느림
- GPU 확인: `nvidia-smi`
- 배치 크기 조정
- LoRA rank 조정 (r 값 감소)

### 모델이 제대로 학습되지 않음
- 학습률 조정 (예: 1e-4)
- 에폭 수 증가
- 데이터 품질 확인

## 🧪 모델 성능 테스트 질문

파인튜닝된 모델의 성능을 평가하기 위한 테스트 질문들입니다. 다양한 주제와 난이도를 포함하여 모델의 이해도와 답변 품질을 확인할 수 있습니다.

### 📊 난이도별 테스트 질문

#### 1. 기본 수준 - 간단한 개념 질문
```
ALD 공정이 뭐야?
```
**평가 포인트**: 기본 개념 이해, 간결한 설명 능력

#### 2. 중급 수준 - 구체적인 장비/공정 파라미터 질문
```
MFC가 뭔지 설명해줘.
```
**평가 포인트**: 장비 기능 설명, 전문 용어 사용

#### 3. 중상급 수준 - 공정 조건과 영향 질문
```
ALD 공정에서 웨이퍼 온도가 너무 높으면 어떻게 되나요?
```
**평가 포인트**: 공정 파라미터와 결과의 인과관계 이해

#### 4. 고급 수준 - 복합적 문제 해결 질문
```
ALD 공정에서 필름 두께가 웨이퍼 중앙과 가장자리에서 다르게 나오는데, 원인과 해결 방법을 알려줘.
```
**평가 포인트**: 복합 문제 분석, 해결책 제시 능력

### 🎯 주제별 테스트 질문

#### ALD 기본 개념
- "ALD 공정이 뭐야?"
- "ALD와 CVD의 차이점은?"

#### Precursor (원재료)
- "Precursor는 뭐야?"
- "Precursor 주입 시간이 너무 짧으면 어떻게 되나요?"

#### Purge (정리 단계)
- "Purge 단계의 역할은?"
- "Purge 시간이 부족하면 어떤 문제가 생기나요?"

#### MFC (유량 제어)
- "MFC가 뭔지 설명해줘."
- "MFC 유량이 불안정하면 ALD 공정에 어떤 영향을 미치나요?"

#### Flow (가스 유량)
- "가스 Flow가 ALD 공정에 미치는 영향은?"
- "유량 오차가 크면 무엇을 의심해야 하나요?"

#### Valve (밸브)
- "Valve 압력 보정이 제대로 안 되면 어떻게 되나요?"
- "APC Valve의 역할은?"

#### 압력 (Pressure)
- "챔버 압력이 ALD 공정에 미치는 영향은?"
- "압력 overshoot이 발생하면 어떻게 되나요?"

#### 챔버 (Chamber)
- "챔버 내부 반응 균일도를 높이려면?"
- "챔버 온도 분포가 불균일하면?"

#### 웨이퍼 (Wafer)
- "웨이퍼 표면 상태가 ALD 공정에 미치는 영향은?"
- "웨이퍼 온도가 너무 높으면?"

#### VG12/VG13 (진공 게이지)
- "VG12와 VG13의 차이점은?"
- "VG12에서 압력 overshoot이 감지되면?"

#### 로드락 (Loadlock)
- "로드락의 역할은?"
- "로드락 압력이 불안정하면?"

#### 플라즈마 (Plasma)
- "플라즈마 ALD의 장점은?"
- "플라즈마가 낮은 온도에서 박막 형성을 가능하게 하는 이유는?"

### ✅ 평가 기준

각 질문에 대해 다음을 확인하세요:

1. **정확성**: 문서에 기반한 정확한 정보 제공
2. **완성도**: 질문에 대한 충분한 답변 제공
3. **전문성**: 전문 용어의 적절한 사용
4. **구조화**: 논리적이고 구조화된 답변
5. **한국어**: 자연스러운 한국어 표현

### 📝 테스트 시나리오

1. **기본 기능 테스트**: 간단한 질문으로 기본 동작 확인
2. **주제별 테스트**: 각 주제별로 2-3개 질문으로 깊이 확인
3. **복합 문제 테스트**: 여러 요소가 관련된 복잡한 질문으로 종합 능력 확인
4. **경계 케이스 테스트**: 문서에 없는 내용에 대한 적절한 응답 확인

