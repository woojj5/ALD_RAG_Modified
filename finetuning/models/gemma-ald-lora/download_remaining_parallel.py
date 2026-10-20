#!/usr/bin/env python3
from huggingface_hub import hf_hub_download
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

token = os.getenv("HUGGINGFACE_TOKEN")
repo_id = "google/gemma-3-27b-it"
cache_dir = "./models--google--gemma-3-27b-it"

missing_files = [
    "model-00002-of-00012.safetensors",
    "model-00004-of-00012.safetensors",
    "model-00008-of-00012.safetensors",
    "model-00009-of-00012.safetensors"
]

def download_file(filename):
    try:
        print(f"[시작] {filename}")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            cache_dir=cache_dir
        )
        print(f"[완료] {filename}")
        return filename, True
    except Exception as e:
        print(f"[실패] {filename}: {e}")
        return filename, False

print(f"[+] {len(missing_files)}개 파일 병렬 다운로드 시작...\n")

# 최대 2개 파일 동시 다운로드 (서버 부하 방지)
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(download_file, f): f for f in missing_files}
    
    for future in as_completed(futures):
        filename, success = future.result()
        if success:
            print(f"✓ {filename} 완료\n")

print("\n[+] 모든 파일 다운로드 완료!")
