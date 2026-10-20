# ⚡ 빠른 실행 가이드

## 🎯 한 줄로 실행하기

```bash
./scripts/server/run_servers.sh
```

그리고 브라우저에서 **http://localhost:8080** 접속!

---

## 📋 단계별 실행

### 1️⃣ 백엔드 시작 (터미널 1)
```bash
cd /home/keti_spark1/ald-rag-lab
source torch-env/bin/activate
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2️⃣ 프론트엔드 시작 (터미널 2)
```bash
cd /home/keti_spark1/ald-rag-lab/frontend
python3 -m http.server 8080 --bind 0.0.0.0
```

### 3️⃣ 브라우저 접속
- **프론트엔드**: http://localhost:8080
- **백엔드 API**: http://localhost:8000

---

## 🛑 종료하기

```bash
./scripts/server/stop_servers.sh
```

또는

```bash
Ctrl + C  # 각 터미널에서
```

---

## ⚠️ 문제 발생 시

### 포트가 이미 사용 중
```bash
killall uvicorn
killall python3
```

### 서버 상태 확인
```bash
lsof -i :8000  # 백엔드
lsof -i :8080  # 프론트엔드
```

---

**더 자세한 내용은 `실행방법.md` 파일을 참고하세요!**

