import os
from huggingface_hub import HfFolder

print("=== 토큰 확인 ===")
hf_token_env = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
hf_token_file = None
try:
    hf_token_file = HfFolder.get_token()
except:
    pass

print(f"HUGGINGFACE_TOKEN 환경변수: {'설정됨' if os.getenv('HUGGINGFACE_TOKEN') else '없음'}")
print(f"HF_TOKEN 환경변수: {'설정됨' if os.getenv('HF_TOKEN') else '없음'}")
print(f"huggingface-cli 저장 토큰: {'있음' if hf_token_file else '없음'}")

if hf_token_env:
    print(f"\n현재 사용 중인 토큰: 환경변수 ({hf_token_env[:10]}...)")
elif hf_token_file:
    print(f"\n현재 사용 중인 토큰: huggingface-cli 저장 토큰 ({hf_token_file[:10]}...)")
else:
    print("\n토큰이 설정되지 않았습니다.")
