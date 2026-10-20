#!/bin/bash
# Gemma Fine-tuning 실행 스크립트 - Qwen과 동일한 설정

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "Gemma Fine-tuning 실행 (Qwen과 동일한 설정)"
echo "=========================================="

# 1단계: 학습 데이터 확인 (Qwen과 동일한 데이터 사용)
echo ""
echo "[1/3] 학습 데이터 확인 중..."
if [ ! -f "$PROJECT_DIR/finetuning/data/train.jsonl" ] || [ ! -f "$PROJECT_DIR/finetuning/data/eval.jsonl" ]; then
    echo "[!] 학습 데이터가 없습니다. 데이터를 생성합니다..."
    python3 finetuning/scripts/prepare_finetuning_data.py
    
    if [ $? -ne 0 ]; then
        echo "[!] 학습 데이터 생성 실패"
        exit 1
    fi
else
    echo "[+] 학습 데이터가 이미 존재합니다 (Qwen과 동일한 데이터 사용)"
fi

# 2단계: Fine-tuning 실행
echo ""
echo "[2/3] Gemma Fine-tuning 실행 중..."
echo "주의: GPU가 필요하며, 시간이 소요될 수 있습니다."
echo "설정: Qwen과 동일한 하이퍼파라미터 사용"
echo "  - 데이터셋: 동일"
echo "  - LoRA: r=32, alpha=32, dropout=0.05"
echo "  - Epochs: 3, Batch size: 4, Learning rate: 1e-4"
echo ""

python3 finetuning/scripts/finetune_gemma.py \
  --train_file "$PROJECT_DIR/finetuning/data/train.jsonl" \
  --eval_file "$PROJECT_DIR/finetuning/data/eval.jsonl" \
  --output_dir "$PROJECT_DIR/finetuning/models/gemma-ald-lora" \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --save_inference_only

if [ $? -ne 0 ]; then
    echo "[!] Fine-tuning 실패"
    exit 1
fi

# 3단계: 완료 메시지
echo ""
echo "[3/3] 완료!"
echo ""
echo "Fine-tuned Gemma 모델이 저장되었습니다:"
echo "  $PROJECT_DIR/finetuning/models/gemma-ald-lora"
echo ""
echo "사용 방법:"
echo "  rag_core.py에서 다음 설정:"
echo "    LLM_MODEL_NAME = 'google/gemma-3-27b-it'"
echo "    FINETUNED_MODEL_PATH = BASE_DIR / 'finetuning' / 'models' / 'gemma-ald-lora'"

