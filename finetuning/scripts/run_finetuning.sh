#!/bin/bash
# Fine-tuning 실행 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "Fine-tuning 실행"
echo "=========================================="

# 1단계: 학습 데이터 생성 (개선된 버전)
echo ""
echo "[1/3] 개선된 학습 데이터 생성 중..."
python3 finetuning/scripts/prepare_finetuning_data.py

if [ $? -ne 0 ]; then
    echo "[!] 학습 데이터 생성 실패"
    exit 1
fi

# 2단계: Fine-tuning 실행
echo ""
echo "[2/3] Fine-tuning 실행 중..."
echo "주의: GPU가 필요하며, 2-4시간이 소요될 수 있습니다."
echo ""
echo "추론용 모델만 저장하려면 --save_inference_only 옵션을 사용하세요."
echo "  (optimizer 상태는 저장하지 않아 파일 크기가 작아집니다)"

python3 finetuning/scripts/finetune_llama.py \
  --train_file "$PROJECT_DIR/finetuning/data/train.jsonl" \
  --eval_file "$PROJECT_DIR/finetuning/data/eval.jsonl" \
  --output_dir "$PROJECT_DIR/finetuning/models/qwen-ald-lora" \
  --num_epochs 3 \
  --batch_size 4 \
  --save_inference_only

if [ $? -ne 0 ]; then
    echo "[!] Fine-tuning 실패"
    exit 1
fi

# 3단계: 완료 메시지
echo ""
echo "[3/3] 완료!"
echo ""
echo "Fine-tuned 모델이 저장되었습니다:"
echo "  $PROJECT_DIR/finetuning/models/qwen-ald-lora"
echo ""
echo "사용 방법:"
echo "  rag_core.py에서 다음 설정:"
echo "    FINETUNED_MODEL_PATH = BASE_DIR / 'finetuning' / 'models' / 'qwen-ald-lora'"

