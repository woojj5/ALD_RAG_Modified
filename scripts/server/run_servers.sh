#!/bin/bash
# 백엔드와 프론트엔드 서버를 한 번에 실행하는 스크립트

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "🚀 ALD RAG 서버 시작"
echo "=========================================="
echo ""

# 기존 서버 종료
echo "[1/4] 기존 서버 종료 중..."
killall uvicorn 2>/dev/null
killall python3 2>/dev/null
sleep 2

# 백엔드 서버 시작
echo "[2/4] 백엔드 서버 시작 중... (포트 8000)"
if [ -d "torch-env" ]; then
    source torch-env/bin/activate
    cd backend
    nohup uvicorn app:app --host 0.0.0.0 --port 8000 --reload > /tmp/backend.log 2>&1 &
    BACKEND_PID=$!
    echo "   ✅ 백엔드 PID: $BACKEND_PID"
    cd ..
else
    echo "   ❌ 오류: torch-env 가상환경을 찾을 수 없습니다!"
    exit 1
fi

sleep 3

# 프론트엔드 서버 시작
echo "[3/4] 프론트엔드 서버 시작 중... (포트 8080)"
cd frontend
nohup python3 -m http.server 8080 --bind 0.0.0.0 > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   ✅ 프론트엔드 PID: $FRONTEND_PID"
cd ..

sleep 2

# 서버 상태 확인
echo "[4/4] 서버 상태 확인 중..."
sleep 1

if curl -s http://localhost:8000 > /dev/null; then
    echo "   ✅ 백엔드 서버 정상 작동 중"
else
    echo "   ⚠️  백엔드 서버 응답 확인 실패 (아직 시작 중일 수 있음)"
fi

if curl -s http://localhost:8080 > /dev/null; then
    echo "   ✅ 프론트엔드 서버 정상 작동 중"
else
    echo "   ⚠️  프론트엔드 서버 응답 확인 실패 (아직 시작 중일 수 있음)"
fi

echo ""
echo "=========================================="
echo "✅ 서버 시작 완료!"
echo "=========================================="
echo ""
echo "📌 접속 주소:"
echo "   프론트엔드: http://localhost:8080"
echo "   백엔드 API: http://localhost:8000"
echo "   API 문서:   http://localhost:8000/docs"
echo ""
echo "📋 서버 정보:"
echo "   백엔드 PID:  $BACKEND_PID (로그: /tmp/backend.log)"
echo "   프론트엔드 PID: $FRONTEND_PID (로그: /tmp/frontend.log)"
echo ""
echo "🛑 서버 종료 방법:"
echo "   ./stop_servers.sh"
echo "   또는"
echo "   killall uvicorn && killall python3"
echo ""
echo "=========================================="

