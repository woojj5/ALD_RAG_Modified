#!/bin/bash
# FastAPI 서버 시작 스크립트
# 모든 네트워크 인터페이스(0.0.0.0)에서 접근 가능하도록 실행

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

# 가상환경 활성화 (torch-env)
if [ -d "torch-env" ]; then
    source torch-env/bin/activate
    echo "[+] torch-env 가상환경 활성화됨"
else
    echo "[!] 경고: torch-env 가상환경을 찾을 수 없습니다."
fi

# 현재 IP 주소 확인
echo "=========================================="
echo "서버 시작 정보"
echo "=========================================="
echo "현재 IP 주소:"
ifconfig | grep -E "inet 192\.168\.|inet 10\.|inet 172\." | grep -v "127.0.0.1" | head -1 | awk '{print "  " $2}'
echo ""
echo "접속 가능한 주소:"
echo "  - 로컬: http://127.0.0.1:8000"
echo "  - 네트워크: http://$(hostname -I | awk '{print $1}'):8000"
echo "  - Swagger UI: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "=========================================="
echo ""

# FastAPI 서버 시작 (모든 인터페이스에서 접근 가능)
echo "[+] FastAPI 서버 시작 중..."
echo "[+] 종료하려면 Ctrl+C를 누르세요"
echo ""

cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

