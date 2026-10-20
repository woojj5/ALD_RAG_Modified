#!/usr/bin/env python3
"""
Gemma 3 27B Base Model 다운로드 스크립트

이 스크립트는 Hugging Face에서 Gemma 3 27B base model을 다운로드합니다.
qwen-ald-lora와 동일한 구조로 비교 가능하도록 구성되었습니다.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "finetuning" / "models" / "gemma-ald-lora"

# Gemma 모델 이름
# Instruction-tuned 버전 사용 (qwen과 동일하게)
BASE_MODEL_NAME = "google/gemma-3-27b-it"

def download_base_model():
    """
    Gemma base model을 다운로드합니다.
    
    주의사항:
    1. Hugging Face 토큰이 필요할 수 있습니다 (gated model인 경우)
    2. 모델 크기가 약 50GB이므로 충분한 디스크 공간이 필요합니다
    3. 다운로드 시간이 오래 걸릴 수 있습니다
    4. qwen-ald-lora와 동일한 구조로 비교 가능하도록 구성
    """
    print(f"[+] Base Model 다운로드 시작: {BASE_MODEL_NAME}")
    print(f"[+] 저장 경로: {MODEL_DIR}")
    print(f"[+] 참고: qwen-ald-lora와 동일한 구조로 구성됩니다")
    
    # Hugging Face 토큰 확인 (환경변수 또는 huggingface-cli 로그인 토큰)
    # 여러 환경변수 이름 확인 (HUGGINGFACE_TOKEN, HF_TOKEN, HUGGINGFACE_HUB_TOKEN)
    hf_token = (os.getenv("HUGGINGFACE_TOKEN") or 
                os.getenv("HF_TOKEN") or 
                os.getenv("HUGGINGFACE_HUB_TOKEN"))
    
    # huggingface-cli로 로그인한 토큰도 확인
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
            if hf_token:
                print("[+] huggingface-cli로 로그인한 토큰을 사용합니다.")
        except Exception:
            pass
    
    if hf_token:
        print("[+] Hugging Face 토큰이 설정되어 있습니다.")
        print(f"[+] 토큰 앞 10자리: {hf_token[:10]}...")
    else:
        print("[!] Hugging Face 토큰이 설정되지 않았습니다.")
        print("[!] gated model인 경우 토큰이 필요합니다.")
        print("[!] 다음 중 하나를 사용하세요:")
        print("[!]   1. export HUGGINGFACE_TOKEN='your_token_here'")
        print("[!]   2. export HF_TOKEN='your_token_here'")
        print("[!]   3. huggingface-cli login")
        print("[!] 토큰은 https://huggingface.co/settings/tokens 에서 발급받을 수 있습니다.")
    
    # 모델 디렉토리 생성
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Tokenizer 다운로드
        print("[+] Tokenizer 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            token=hf_token,
            cache_dir=str(MODEL_DIR)
        )
        print("[+] Tokenizer 다운로드 완료")
        
        # 모델 다운로드
        print("[+] Base Model 다운로드 중... (시간이 오래 걸릴 수 있습니다)")
        print("[+] 모델 크기: 약 50GB")
        print("[+] 참고: 이 모델은 base model입니다. Fine-tuning 후 LoRA adapter가 생성됩니다.")
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            token=hf_token,
            cache_dir=str(MODEL_DIR),
            torch_dtype=torch.float16,  # 메모리 절약을 위해 float16 사용
            device_map="auto"  # 자동으로 GPU/CPU 할당
        )
        
        print("[+] Base Model 다운로드 완료!")
        print(f"[+] 모델이 다음 경로에 저장되었습니다: {MODEL_DIR}")
        print(f"[+] 다음 단계: Fine-tuning을 통해 LoRA adapter를 생성하세요")
        print(f"[+] Fine-tuning 후 adapter는 이 디렉토리에 저장됩니다 (qwen-ald-lora와 동일한 구조)")
        
        # 모델 정보 저장 (qwen-ald-lora의 README.md와 유사한 형식)
        info_file = MODEL_DIR / "BASE_MODEL_INFO.txt"
        with open(info_file, "w", encoding="utf-8") as f:
            f.write(f"Base Model Name: {BASE_MODEL_NAME}\n")
            f.write(f"Model Path: {MODEL_DIR}\n")
            f.write(f"Model Size: ~50GB\n")
            f.write(f"Download Date: {Path(__file__).stat().st_mtime}\n")
            f.write(f"\n")
            f.write(f"참고:\n")
            f.write(f"- 이 디렉토리는 qwen-ald-lora와 동일한 구조로 구성됩니다\n")
            f.write(f"- Fine-tuning 후 LoRA adapter 파일들이 이 디렉토리에 저장됩니다\n")
            f.write(f"- Base model은 Hugging Face 캐시에 저장되며, 이 디렉토리에는 adapter만 저장됩니다\n")
        
        print(f"[+] 모델 정보가 저장되었습니다: {info_file}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[!] 다운로드 중 오류 발생: {e}")
        print("\n[!] 다음을 확인하세요:")
        print("    1. 인터넷 연결 상태")
        print("    2. 디스크 공간 (최소 60GB 이상 필요)")
        print("    3. Hugging Face 토큰 권한")
        
        # 403 Forbidden 에러인 경우 상세 안내
        if "403" in error_msg or "Forbidden" in error_msg or "public gated repositories" in error_msg.lower():
            print("\n" + "="*70)
            print("[!] 403 Forbidden 에러 - 토큰 권한 문제")
            print("="*70)
            print("\n[!] Fine-grained token에 'Public gated repositories' 접근 권한이 필요합니다.")
            print("\n[!] 해결 방법 (선택 1): Fine-grained token 권한 수정")
            print("    1. https://huggingface.co/settings/tokens 접속")
            print("    2. 'spark-gemma' 토큰 클릭하여 편집")
            print("    3. 'Repository access' 섹션에서:")
            print("       - 'Public gated repositories' 체크박스 활성화")
            print("       또는")
            print("       - 'All repositories' 선택")
            print("    4. 'Save' 클릭")
            print("\n[!] 해결 방법 (선택 2): Classic token 사용 (권장)")
            print("    1. https://huggingface.co/settings/tokens 접속")
            print("    2. 'New token' 클릭")
            print("    3. 'Classic' 선택")
            print("    4. 'Read' 권한 선택")
            print("    5. 토큰 이름 입력 (예: gemma-classic)")
            print("    6. 'Generate token' 클릭 후 토큰 복사")
            print("    7. 다음 명령어 실행:")
            print("       export HUGGINGFACE_TOKEN='복사한_토큰'")
            print("       또는")
            print("       huggingface-cli login")
            print("\n[!] 추가 확인사항:")
            print("    - 브라우저에서 https://huggingface.co/google/gemma-3-27b-it 접속")
            print("    - 'Agree and access repository' 버튼 클릭하여 라이선스 승인")
            print("    - 이미 승인했다면 토큰 권한만 수정하면 됩니다")
            print("="*70)
        
        # 모델 이름 확인
        if "gemma-2-27b" in error_msg.lower():
            print("\n[!] 경고: gemma-2-27b-it이 감지되었습니다!")
            print(f"[!] 현재 설정된 모델: {BASE_MODEL_NAME}")
            print("[!] 코드를 확인하여 모든 gemma-2-27b-it 참조를 gemma-3-27b-it으로 변경하세요.")
        
        raise


if __name__ == "__main__":
    download_base_model()

