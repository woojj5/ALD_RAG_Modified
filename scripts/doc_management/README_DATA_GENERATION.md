# 📚 docs_ald.json 데이터 생성 가이드

`docs_ald.json`에 유의미한 데이터를 추가하는 여러 방법을 제공합니다.

## 🚀 빠른 시작

### 1. LLM 기반 생성 (추천)
기존 데이터를 학습하여 유사한 스타일의 새 문장을 생성합니다.

```bash
# ALD 키워드에 5개 문장 생성
python generate_docs.py llm --keyword ALD --count 5

# Precursor 키워드에 3개 문장 생성
python generate_docs.py llm --keyword Precursor --count 3
```

**장점:**
- 기존 데이터의 스타일과 톤을 유지
- 실제 공정 지식을 반영한 문장 생성
- 키워드별 특성에 맞는 내용 생성

**주의사항:**
- 기존 데이터가 최소 1개 이상 있어야 함
- LLM 모델이 로드되어 있어야 함 (첫 실행 시 시간 소요)

### 2. 템플릿 기반 생성
미리 정의된 템플릿을 사용하여 데이터를 생성합니다.

```bash
# MFC 키워드에 3개 문장 생성
python generate_docs.py template --keyword MFC --count 3

# Flow 키워드에 5개 문장 생성
python generate_docs.py template --keyword Flow --count 5
```

**장점:**
- 빠른 생성
- 일관된 형식
- LLM 모델 불필요

**단점:**
- 템플릿이 정의된 키워드만 가능
- 다양성이 제한적

### 3. 수동 추가
`manage_docs.py`를 사용하여 직접 추가합니다.

```bash
# 인터랙티브 모드
python manage_docs.py add

# 명령줄 모드
python manage_docs.py add --keyword "새키워드" --text "새로운 문장 내용"
```

## 📋 지원되는 키워드 (템플릿)

현재 템플릿이 정의된 키워드:
- ALD
- Precursor
- Purge
- MFC
- Flow
- 플라즈마
- 압력
- 챔버
- VG12
- VG13
- 로드락
- Valve
- 웨이퍼

## 💡 데이터 생성 전략

### 단계별 접근

1. **기존 키워드 확장**
   ```bash
   # 각 키워드마다 3-5개씩 추가
   python generate_docs.py llm --keyword ALD --count 5
   python generate_docs.py llm --keyword Precursor --count 3
   python generate_docs.py llm --keyword Purge --count 3
   ```

2. **새 키워드 추가**
   - 먼저 수동으로 1-2개 문장 추가
   ```bash
   python manage_docs.py add --keyword "새키워드" --text "첫 번째 문장"
   ```
   - 그 다음 LLM으로 확장
   ```bash
   python generate_docs.py llm --keyword "새키워드" --count 5
   ```

3. **다양성 확보**
   - LLM 생성: 실제 공정 지식 반영
   - 템플릿 생성: 구조화된 정보
   - 수동 추가: 특수 케이스나 경험 기반 정보

## 🎯 유의미한 데이터 작성 팁

### 좋은 문장의 특징
- ✅ 구체적: "ALD 공정은 200-300℃에서 수행된다"
- ✅ 명확: "MFC는 0-1000 sccm 범위에서 1% 정밀도로 제어한다"
- ✅ 실용적: "유량 오차가 5% 이상이면 MFC 이상을 의심해야 한다"
- ✅ 독립적: 각 문장이 단독으로 의미를 가짐

### 피해야 할 문장
- ❌ 너무 일반적: "ALD는 좋은 공정이다"
- ❌ 모호함: "압력이 중요하다"
- ❌ 중복: 기존 문장과 거의 동일한 내용

## 📊 데이터 확인

생성된 데이터를 확인하려면:

```bash
# 키워드별 그룹화 보기
python manage_docs.py group

# 통계 보기
python manage_docs.py stats
```

## 🔄 워크플로우 예시

```bash
# 1. 현재 상태 확인
python manage_docs.py stats

# 2. ALD 키워드 확장
python generate_docs.py llm --keyword ALD --count 5

# 3. 새 키워드 추가 (예: Temperature)
python manage_docs.py add --keyword Temperature --text "ALD 공정 온도는 150-400℃ 범위에서 설정된다."

# 4. 새 키워드 확장
python generate_docs.py llm --keyword Temperature --count 4

# 5. 결과 확인
python manage_docs.py group
```

## 🛠️ 고급 사용법

### 배치 생성 스크립트 예시

```bash
#!/bin/bash
# 모든 키워드에 3개씩 추가

keywords=("ALD" "Precursor" "Purge" "MFC" "Flow" "플라즈마" "압력")

for keyword in "${keywords[@]}"; do
    echo "Processing $keyword..."
    python generate_docs.py llm --keyword "$keyword" --count 3
done
```

## ⚠️ 주의사항

1. **품질 검토**: 생성된 문장은 반드시 검토 후 추가
2. **중복 확인**: 기존 문장과 중복되지 않는지 확인
3. **정확성**: 기술적 정확성을 확인
4. **백업**: 대량 추가 전 백업 권장

## 📝 향후 개선 사항

- [ ] 웹 검색 기반 자동 수집
- [ ] PDF/문서 파싱 기능
- [ ] 자동 중복 제거
- [ ] 품질 점수 평가
- [ ] 배치 처리 최적화

