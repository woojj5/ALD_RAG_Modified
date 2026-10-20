#!/usr/bin/env python3
"""
Gemma Base Model 테스트 스크립트

다운로드한 Gemma base model이 제대로 작동하는지 테스트합니다.
qwen-ald-lora와 동일한 방식으로 테스트합니다.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import os

# 모델 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "finetuning" / "models" / "gemma-ald-lora"
BASE_MODEL_NAME = "google/gemma-3-27b-it"  # 다운로드한 모델 이름과 동일하게

def test_base_model():
    """
    Gemma base model을 로드하고 간단한 테스트를 수행합니다.
    qwen-ald-lora와 동일한 방식으로 테스트합니다.
    """
    print("[+] Gemma Base Model 테스트 시작")
    print(f"[+] Base Model: {BASE_MODEL_NAME}")
    print(f"[+] 경로: {MODEL_DIR}")
    
    # 디바이스 확인
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print(f"[+] GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"[+] GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("[!] GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        print("[!] CPU 모드는 매우 느릴 수 있습니다.")
    
    # Hugging Face 토큰 (환경변수 또는 huggingface-cli 로그인 토큰)
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    # huggingface-cli로 로그인한 토큰도 확인
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            pass
    
    try:
        # Tokenizer 로드
        print("[+] Tokenizer 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            token=hf_token,
            cache_dir=str(MODEL_DIR)
        )
        
        # Gemma의 경우 pad_token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("[+] Tokenizer 로드 완료")
        
        # 모델 로드
        print("[+] Base Model 로드 중... (시간이 걸릴 수 있습니다)")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            token=hf_token,
            cache_dir=str(MODEL_DIR),
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None
        )
        
        if device.type == "cpu":
            model = model.to(device)
        
        print("[+] Base Model 로드 완료")
        
        # 테스트 프롬프트 (qwen 테스트와 유사한 형식)
        test_prompt = "반도체 ALD 공정에 대해 간단히 설명해주세요."
        print(f"\n[+] 테스트 프롬프트: {test_prompt}\n")
        
        # Gemma chat template 사용 (qwen과 유사)
        messages = [
            {"role": "user", "content": test_prompt}
        ]
        
        # Chat template 적용
        if hasattr(tokenizer, "apply_chat_template"):
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            full_prompt = test_prompt
        
        # 토큰화
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        # 생성
        print("[+] 답변 생성 중...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[+] 생성된 답변:\n{response}\n")
        
        print("[+] Base Model 테스트 완료!")
        print("[+] 다음 단계: Fine-tuning을 통해 LoRA adapter를 생성하세요")
        
    except Exception as e:
        print(f"[!] 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_base_model()

