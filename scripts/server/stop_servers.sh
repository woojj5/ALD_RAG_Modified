#!/bin/bash
# 실행 중인 서버 종료 스크립트

echo "=========================================="
echo "🛑 ALD RAG 서버 종료"
echo "=========================================="
echo ""

echo "[1/2] 백엔드 서버 종료 중..."
killall uvicorn 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ 백엔드 서버 종료 완료"
else
    echo "   ℹ️  실행 중인 백엔드 서버가 없습니다"
fi

echo "[2/2] 프론트엔드 서버 종료 중..."
killall python3 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ 프론트엔드 서버 종료 완료"
else
    echo "   ℹ️  실행 중인 프론트엔드 서버가 없습니다"
fi

sleep 1

# 포트 확인
if lsof -i :8000 > /dev/null 2>&1; then
    echo "   ⚠️  포트 8000이 아직 사용 중입니다"
fi

if lsof -i :8080 > /dev/null 2>&1; then
    echo "   ⚠️  포트 8080이 아직 사용 중입니다"
fi

echo ""
echo "=========================================="
echo "✅ 서버 종료 완료!"
echo "=========================================="

