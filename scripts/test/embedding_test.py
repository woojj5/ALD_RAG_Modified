from sentence_transformers import SentenceTransformer  # type: ignore
import numpy as np  # type: ignore

# 1) 임베딩 모델 불러오기
# 필요하면 여기만 "BAAI/bge-m3" 로 바꾸면 됨
model_name = "thenlper/gte-small"  # 또는 "BAAI/bge-m3"
emb_model = SentenceTransformer(model_name)

# 2) 검색 대상 문장들
docs = [
    "ALD 공정은 Precursor A와 B를 번갈아 주입해 원자층 단위로 박막을 쌓는 공정이다.",
    "ALD는 Surface self-limiting 반응을 이용해 한 사이클당 일정 두께만큼만 증착된다.",
    "Precursor는 박막을 만들기 위한 원재료 기체이다.",
    "Precursor A는 웨이퍼 표면에 한 층만 흡착되면 반응이 멈춘다.",
    "Purge는 챔버 내부의 남아있는 기체를 제거하는 정리 단계이다.",
    "Purge는 A와 B가 공중에서 섞여 이상 반응을 일으키지 않도록 한다.",
    "플라즈마 ALD는 라디칼을 이용해 낮은 온도에서도 높은 반응성을 확보한다.",
    "플라즈마를 사용하면 박막 밀도와 특성이 향상된다.",
    "MFC는 가스 유량을 정밀하게 제어하는 장치이다.",
    "유량은 sccm 단위로 표시되며 가스가 흐르는 양을 의미한다.",
    "APC 밸브는 챔버 압력을 일정하게 유지하기 위해 열리고 닫힌다.",
    "챔버는 반도체 공정이 실제로 일어나는 밀폐된 공간이다.",
    "VG12는 공정 챔버의 실제 압력을 측정하는 진공 게이지이다.",
    "VG13은 로드락 또는 전단부 압력을 측정하는 센서이다.",
    "로드락 챔버는 웨이퍼를 외부에서 내부로 이동시키기 위한 진공 전환 공간이다.",
    "압력 Overshoot는 가스가 과도하게 주입되거나 APC 제어가 늦어 발생한다.",
    "유량 Set 대비 실제 유량 오차가 크면 MFC 이상을 의심할 수 있다.",
    "Valve stuck-open 패턴은 밸브가 닫히지 않고 계속 열린 상태로 고착된 상황이다.",
    "반도체 웨이퍼는 여러 공정을 반복해 회로를 형성하는 얇은 실리콘 판이다.",
    "ALD 공정 조건은 시간 제어보다 표면 반응 특성에 의해 결정된다."
]


# 3) 문서 임베딩 만들기
doc_embs = emb_model.encode(docs, normalize_embeddings=True)
print("문서 임베딩 shape:", doc_embs.shape)


# 4) 검색 함수
def search(query: str, top_k: int = 3):
    # 4-1) 쿼리 임베딩
    q_emb = emb_model.encode([query], normalize_embeddings=True)[0]

    # 4-2) 코사인 유사도 계산
    scores = np.dot(doc_embs, q_emb)

    # 4-3) 높은 순 정렬
    idx_sorted = np.argsort(-scores)[:top_k]

    print("\n==============================")
    print(f"🧐 Query: {query}")
    print("🔎 검색 결과 (유사도 순):")
    for i in idx_sorted:
        print(f"score={scores[i]:.3f} | {docs[i]}")
    print("==============================\n")


# 5) 직접 질문 입력하는 모드
if __name__ == "__main__":
    while True:
        query = input("\n질문 입력 (엔터만 치면 종료): ")
        if not query.strip():
            print("끝냄!")
            break
        search(query, top_k=3)
