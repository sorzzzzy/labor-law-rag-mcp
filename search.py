"""같은 질문을 세 가지 방식으로 검색해 결과를 비교한다.

- bm25   : 키워드 검색. nori로 형태소 분석한 content 필드에서 단어 일치로 찾는다.
- knn    : 벡터 검색. 질문을 bge-m3로 임베딩해서 의미가 가까운 청크를 찾는다.
- hybrid : 두 결과를 RRF(순위 기반 합치기)로 합친다. 직접 구현해서 원리를 확인한다.

사용법
  python search.py "입사한 지 1년 된 직원은 연차가 며칠이야?"   # 질문 하나를 세 방식으로 비교
  python search.py --eval                                        # 준비한 질문 8개로 한 번에 비교
"""
import sys

import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

ES_URL = "http://localhost:9200"
INDEX = "labor_law"
MODEL_NAME = "BAAI/bge-m3"
RRF_K = 60          # RRF 상수. 순위 차이가 점수에 너무 크게 반영되지 않도록 완충한다
CANDIDATES = 20     # 하이브리드에서 각 검색이 가져올 후보 수

es = Elasticsearch(ES_URL)
_model = None


def model():
    """임베딩 모델은 무거우니 처음 쓸 때 한 번만 불러온다."""
    global _model
    if _model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def _hits(resp):
    """Elasticsearch 응답에서 필요한 값만 뽑는다."""
    return [
        {"id": h["_id"], "score": h["_score"], **h["_source"]}
        for h in resp["hits"]["hits"]
    ]


def bm25(query, k=5):
    """키워드 검색: content 필드(nori 분석)에서 BM25 점수가 높은 순."""
    resp = es.search(
        index=INDEX,
        query={"match": {"content": query}},
        size=k,
        source_excludes=["embedding"],
    )
    return _hits(resp)


def knn(query, k=5):
    """벡터 검색: 질문 벡터와 코사인 유사도가 높은 순."""
    # 색인할 때와 같은 모델·같은 정규화로 질문을 벡터로 바꿔야 비교가 의미 있다
    vec = model().encode(query, normalize_embeddings=True).tolist()
    resp = es.search(
        index=INDEX,
        knn={
            "field": "embedding",
            "query_vector": vec,
            "k": k,                 # 최종으로 돌려받을 개수
            "num_candidates": 100,  # HNSW 그래프에서 살펴볼 후보 수 (클수록 정확, 느림)
        },
        size=k,
        source_excludes=["embedding"],
    )
    return _hits(resp)


def hybrid(query, k=5):
    """RRF: 각 검색 결과의 '순위'만 보고 1/(60+순위)를 더한다.

    BM25 점수(상한 없음)와 코사인 유사도(0~1)는 단위가 달라서 그대로 더할 수 없다.
    순위는 두 검색이 같은 기준이라 공평하게 합칠 수 있다.
    """
    scores, docs = {}, {}
    for results in (bm25(query, CANDIDATES), knn(query, CANDIDATES)):
        for rank, doc in enumerate(results, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (RRF_K + rank)
            docs[doc["id"]] = doc
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**docs[i], "score": scores[i]} for i in top]


METHODS = {"bm25": bm25, "knn": knn, "hybrid": hybrid}


def show(query, k=5):
    """질문 하나를 세 방식으로 검색해 나란히 출력한다."""
    print(f"\n질문: {query}")
    for name, fn in METHODS.items():
        print(f"\n[{name}]")
        for rank, d in enumerate(fn(query, k), start=1):
            title = f"({d['title']})" if d.get("title") else ""
            print(f"  {rank}. {d['id']}{title}  score={d['score']:.4f}")


# ──────────────────────────────────────────────────────────────
# 비교용 질문 8개: (질문, 정답 조문)
#   앞의 4개: 조문 번호나 법률 용어를 그대로 쓴 질문 → 키워드 검색이 유리할 것으로 예상
#   뒤의 4개: 일상 표현으로 풀어 쓴 질문 → 벡터 검색이 유리할 것으로 예상
# ──────────────────────────────────────────────────────────────
EVAL_SET = [
    ("제60조 연차 유급휴가", "제60조"),
    ("제76조의2 직장 내 괴롭힘의 금지", "제76조의2"),
    ("해고의 예고", "제26조"),
    ("연장근로에 대한 가산임금", "제56조"),
    ("입사한 지 1년 된 직원은 연차가 며칠이야?", "제60조"),
    ("상사가 계속 모욕적인 말을 하고 따돌리면 법적으로 보호받을 수 있어?", "제76조의2"),
    ("회사가 갑자기 내일부터 나오지 말라고 하면 미리 알려줘야 하는 거 아니야?", "제26조"),
    ("밤늦게까지 야근하면 돈을 더 받을 수 있어?", "제56조"),
]


def rank_of(results, article):
    """정답 조문이 결과에서 몇 위에 나왔는지. 없으면 None."""
    for rank, d in enumerate(results, start=1):
        if d["article"] == article:
            return rank
    return None


def evaluate(k=5):
    """질문마다 세 방식에서 정답 조문이 몇 위에 나왔는지 표로 출력한다."""
    print(f"정답 조문의 순위 (상위 {k}개 안에 없으면 -)\n")
    print(f"{'질문':<46} {'정답':<9} {'bm25':>5} {'knn':>5} {'hybrid':>7}")
    hits = {name: 0 for name in METHODS}
    for query, answer in EVAL_SET:
        ranks = {}
        for name, fn in METHODS.items():
            r = rank_of(fn(query, k), answer)
            ranks[name] = r
            hits[name] += r is not None
        cells = [str(ranks[n]) if ranks[n] else "-" for n in METHODS]
        short = query if len(query) <= 24 else query[:23] + "…"
        print(f"{short:<46} {answer:<9} {cells[0]:>5} {cells[1]:>5} {cells[2]:>7}")
    total = len(EVAL_SET)
    print(f"\n상위 {k}개 안에 정답이 든 질문 수: "
          + ", ".join(f"{n} {hits[n]}/{total}" for n in METHODS))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        evaluate()
    else:
        show(" ".join(sys.argv[1:]) or "입사한 지 1년 된 직원은 연차가 며칠이야?")
