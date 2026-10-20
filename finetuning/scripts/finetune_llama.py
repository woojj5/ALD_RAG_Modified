#!/usr/bin/env python3
# LLaMA Fine-tuning 스크립트 (LoRA + QLoRA 4비트 양자화 및 메모리 최적화 적용)

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# 파이토치 메모리 단편화(Fragmentation) 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 프로젝트 루트 경로 추가
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from datasets import load_dataset  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training  # type: ignore
    import torch  # type: ignore
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"[!] 필요한 패키지가 설치되지 않았습니다: {e}")
    print("[!] 다음 명령어로 설치하세요:")
    print("    pip install transformers datasets peft accelerate bitsandbytes")
    DEPENDENCIES_AVAILABLE = False


def load_jsonl(filepath: Path) -> List[Dict[str, str]]:
    """JSONL 파일 로드"""
    data = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_instruction(item: Dict[str, Any], tokenizer) -> str:
    """Qwen2.5 Instruct 형식으로 프롬프트 변환"""
    # 새로운 형식 (messages 기반)
    if "messages" in item:
        messages = item["messages"]
        # chat template 적용
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    # 기존 형식 (하위 호환성)
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")
    
    if instruction and input_text and output:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        return prompt
    
    return ""


def prepare_dataset(train_file: Path, eval_file: Path, tokenizer):
    """데이터셋 준비"""
    
    def tokenize_function(examples):
        # 새로운 형식 (messages 기반) 또는 기존 형식 지원
        prompts = []
        labels_list = []
        
        # messages 형식인지 확인
        if "messages" in examples:
            messages_list = examples["messages"]
            for messages in messages_list:
                # 전체 대화를 프롬프트로 변환
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                prompts.append(full_prompt)
                
                # assistant 응답만 추출하여 labels 생성
                assistant_msg = None
                for msg in messages:
                    if msg.get("role") == "assistant":
                        assistant_msg = msg.get("content", "")
                        break
                labels_list.append(assistant_msg if assistant_msg else "")
        else:
            # 기존 형식 (하위 호환성)
            instructions = examples.get("instruction", [])
            inputs = examples.get("input", [])
            outputs = examples.get("output", [])
            
            for i in range(max(len(instructions), len(inputs), len(outputs))):
                item = {
                    "instruction": instructions[i] if i < len(instructions) else "",
                    "input": inputs[i] if i < len(inputs) else "",
                    "output": outputs[i] if i < len(outputs) else ""
                }
                prompt = format_instruction(item, tokenizer)
                prompts.append(prompt)
                labels_list.append(item.get("output", ""))
        
        # 토크나이징
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=512,  # 메모리 절약을 위해 512로 최적화
            padding="max_length",
            return_tensors="pt"
        )
        
        # Labels 생성: assistant 응답 부분만 학습 대상
        if "messages" in examples:
            labels = []
            for label_text in labels_list:
                if label_text:
                    label_tokens = tokenizer(
                        label_text,
                        truncation=True,
                        max_length=512,
                        add_special_tokens=False
                    )["input_ids"]
                    label_tokens = label_tokens + [tokenizer.pad_token_id] * (512 - len(label_tokens))
                    labels.append(label_tokens[:512])
                else:
                    labels.append([tokenizer.pad_token_id] * 512)
            
            tokenized["labels"] = torch.tensor(labels)
        else:
            tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # 데이터 로드
    train_data = load_jsonl(train_file)
    eval_data = load_jsonl(eval_file)
    
    print(f"[+] 학습 데이터: {len(train_data)}개")
    print(f"[+] 검증 데이터: {len(eval_data)}개")
    
    # 데이터셋 생성
    from datasets import Dataset  # type: ignore
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    # 토크나이징
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="LLaMA Fine-tuning (LoRA + QLoRA)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="기본 모델 이름"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(BASE_DIR / "finetuning" / "data" / "train.jsonl"),
        help="학습 데이터 파일"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=str(BASE_DIR / "finetuning" / "data" / "eval.jsonl"),
        help="검증 데이터 파일"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(BASE_DIR / "finetuning" / "models" / "qwen-ald-lora"),
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="학습 에폭 수"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # VRAM 안정성을 위해 배치 사이즈를 1로 낮춤
        help="배치 크기"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="학습률"
    )
    parser.add_argument(
        "--save_inference_only",
        action="store_true",
        help="추론용 모델만 저장"
    )
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        return
    
    print("[+] Fine-tuning 시작...")
    print(f"  - 모델: {args.model_name}")
    print(f"  - 학습 데이터: {args.train_file}")
    print(f"  - 출력: {args.output_dir}")
    
    # 모델 및 토크나이저 로드 (QLoRA 4비트 양자화 적용)
    print("[+] 모델 로딩 중 (4비트 QLoRA 적용)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 4비트 학습을 위한 peft 준비 작업
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정
    print("[+] LoRA 설정 중...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # VRAM 절약을 위해 16으로 설정
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 데이터셋 준비
    print("[+] 데이터셋 준비 중...")
    train_dataset, eval_dataset = prepare_dataset(
        Path(args.train_file),
        Path(args.eval_file),
        tokenizer
    )
    
    save_only_model = args.save_inference_only
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,         # 배치 사이즈가 1이므로 누적 단계를 늘려 유효 배치 유지
        gradient_checkpointing=True,           # [필수] 메모리 절약을 위한 그래디언트 체킹 활성화
        gradient_checkpointing_kwargs={"use_reentrant": False}, # 최신 PyTorch 경고 대응
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        save_only_model=save_only_model,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("[+] 학습 시작...")
    trainer.train()
    
    print(f"[+] 모델 저장 중: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    if args.save_inference_only:
        print("[+] 체크포인트에서 optimizer 파일 정리 중...")
        output_path = Path(args.output_dir)
        deleted_count = 0
        for checkpoint_dir in output_path.glob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                files_to_remove = [
                    checkpoint_dir / "optimizer.pt",
                    checkpoint_dir / "scheduler.pt",
                    checkpoint_dir / "scaler.pt",
                    checkpoint_dir / "rng_state.pth",
                ]
                for file in files_to_remove:
                    if file.exists():
                        file.unlink()
                        deleted_count += 1
                        print(f"  - 삭제: {checkpoint_dir.name}/{file.name}")
        if deleted_count > 0:
            print(f"[+] 총 {deleted_count}개의 optimizer 파일이 삭제되었습니다")
            
    print("[+] 완료!")


if __name__ == "__main__":
    main()