"""같은 질문을 세 가지 방식으로 검색해 결과를 비교한다.

- bm25   : 키워드 검색. nori로 형태소 분석한 content 필드에서 단어 일치로 찾는다.
- knn    : 벡터 검색. 질문을 bge-m3로 임베딩해서 의미가 가까운 청크를 찾는다.
- hybrid : 두 결과를 RRF(순위 기반 합치기)로 합친다. 직접 구현해서 원리를 확인한다.

사용법
  python search.py "입사한 지 1년 된 직원은 연차가 며칠이야?"   # 질문 하나를 세 방식으로 비교
  python search.py --eval                                        # 준비한 질문 20개로 한 번에 비교
  python search.py --eval --index labor_law_noheader             # 다른 인덱스로 비교 (실험용)
  python search.py --sweep                                       # RRF 설정 조합 비교 (실험용)
  python search.py --holdout                                     # 검증용 질문 8개로 비교 (실험용)
"""
import argparse

import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

ES_URL = "http://localhost:9200"
INDEX = "labor_law"
MODEL_NAME = "BAAI/bge-m3"
RRF_K = 60          # RRF 상수. 순위 차이가 점수에 너무 크게 반영되지 않도록 완충한다
CANDIDATES = 10     # 하이브리드에서 각 검색이 가져올 후보 수
W_BM25 = 0.5        # 하이브리드에서 BM25 쪽 가중치 (kNN은 1). 1보다 작으면 kNN 순위를 더 믿는다
# CANDIDATES와 W_BM25는 처음에 20, 1.0이었다. RRF 조정 실험(--sweep)으로 10, 0.5를 고르고,
# 튜닝에 쓰지 않은 검증용 질문(--holdout)에서도 기존 설정보다 나은 것을 확인한 뒤 바꿨다.

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
    return rrf(bm25(query, CANDIDATES), knn(query, CANDIDATES), k, RRF_K, CANDIDATES, W_BM25)


def rrf(bm25_results, knn_results, k=5, rrf_k=RRF_K, candidates=CANDIDATES, w_bm25=W_BM25):
    """두 검색 결과를 RRF로 합친다. hybrid()와 RRF 조정 실험(--sweep)이 같이 쓴다.

    rrf_k      : RRF 상수. 작을수록 상위 순위에 점수가 몰린다 (1위 1/(k+1)과 2위 1/(k+2)의 차이가 커짐)
    candidates : 각 검색 결과에서 앞쪽 몇 개만 쓸지
    w_bm25     : BM25 쪽 가중치 (kNN은 1로 고정). 1보다 작으면 kNN 순위를 더 믿는다
    """
    scores, docs = {}, {}
    for results, weight in ((bm25_results[:candidates], w_bm25), (knn_results[:candidates], 1.0)):
        for rank, doc in enumerate(results, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0) + weight / (rrf_k + rank)
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
# 비교용 질문 20개: (유형, 질문, 정답)
#   정답이 "제60조"처럼 조 번호면 그 조의 어느 청크든 찾으면 맞힌 것으로,
#   "제60조②"처럼 항까지 있으면 그 항 청크를 찾아야 맞힌 것으로 본다.
#
#   조 번호·제목 : 조 번호나 조 제목을 그대로 쓴 질문
#   법률 용어    : 조문에 나오는 법률 용어로 쓴 질문 → 키워드 검색이 유리할 것으로 예상
#   항 단위      : 정답이 긴 조의 특정 항인 질문 → 항 단위 청킹과 상위 제목의 효과를 보려고
#   일상 표현    : 법률 용어 없이 풀어 쓴 질문 → 벡터 검색이 유리할 것으로 예상
#   주제 섞임    : 여러 주제를 검색어 하나에 넣은 질문 (MCP 실험에서 Claude가 실제로 보낸 검색어 포함)
# ──────────────────────────────────────────────────────────────
EVAL_SET = [
    ("조 번호·제목", "제60조 연차 유급휴가", "제60조"),
    ("조 번호·제목", "제76조의2 직장 내 괴롭힘의 금지", "제76조의2"),
    ("조 번호·제목", "제54조 휴게", "제54조"),
    ("조 번호·제목", "제93조 취업규칙의 작성·신고", "제93조"),
    ("법률 용어", "해고의 예고", "제26조"),
    ("법률 용어", "연장근로에 대한 가산임금", "제56조"),
    ("법률 용어", "휴업수당 지급 기준", "제46조"),
    ("법률 용어", "임금채권의 소멸시효", "제49조"),
    ("항 단위", "계속 근로 기간이 1년 미만인 근로자의 유급휴가", "제60조②"),
    ("항 단위", "연차를 1년 동안 안 쓰면 없어져?", "제60조⑦"),
    ("항 단위", "출산전후휴가는 며칠이야?", "제74조①"),
    ("항 단위", "1주에 연장근로는 최대 몇 시간까지 할 수 있어?", "제53조①"),
    ("일상 표현", "입사한 지 1년 된 직원은 연차가 며칠이야?", "제60조"),
    ("일상 표현", "상사가 계속 모욕적인 말을 하고 따돌리면 법적으로 보호받을 수 있어?", "제76조의2"),
    ("일상 표현", "회사가 갑자기 내일부터 나오지 말라고 하면 미리 알려줘야 하는 거 아니야?", "제26조"),
    ("일상 표현", "밤늦게까지 야근하면 돈을 더 받을 수 있어?", "제56조"),
    ("일상 표현", "점심시간은 법으로 얼마나 보장돼?", "제54조"),
    ("일상 표현", "회사를 그만뒀는데 마지막 월급은 언제까지 받아야 해?", "제36조"),
    ("주제 섞임", "해고의 예고 서면통지 부당해고 구제신청", "제26조"),
    ("주제 섞임", "연장근로 한도와 야간·휴일 가산수당, 보상 휴가제", "제53조"),
]

# ──────────────────────────────────────────────────────────────
# 검증용 질문 8개 (python search.py --holdout)
#   RRF 조정(--sweep)에 쓰지 않은 질문. 위의 20개로 고른 설정이 처음 보는 질문에서도 통하는지 확인한다.
#   튜닝에 쓴 질문으로만 평가하면, 그 질문들에만 잘 맞는 설정을 고를 위험이 있다(과적합).
# ──────────────────────────────────────────────────────────────
EVAL_SET_HOLDOUT = [
    ("법률 용어", "근로조건이 사실과 다를 때 근로계약 즉시 해제", "제19조"),
    ("항 단위", "임신 중인 여성 근로자의 시간외근로 금지", "제74조⑤"),
    ("일상 표현", "근로계약서 쓸 때 꼭 적어야 하는 게 뭐야?", "제17조"),
    ("일상 표현", "하루 기본 근무시간은 몇 시간이야?", "제50조"),
    ("일상 표현", "아기 키우면서 일하는데 수유 시간을 받을 수 있어?", "제75조"),
    ("일상 표현", "일하다 다쳤는데 치료비는 회사가 내줘야 해?", "제78조"),
    ("주제 섞임", "임금 체불 지연이자 금품 청산 임금채권 우선변제", "제37조"),
    ("주제 섞임", "출산전후휴가 육아 시간 생리휴가 태아검진", "제74조의2"),
]

# 항 번호로 쓰이는 원문자. 정답 끝에 이 글자가 있으면 항 단위 정답이다
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def rank_of(results, answer):
    """정답이 결과에서 몇 위에 나왔는지. 없으면 None.

    정답에 항 번호가 붙어 있으면 청크 id("제60조②")로, 아니면 조 번호("제60조")로 비교한다.
    """
    key = "id" if answer[-1] in CIRCLED else "article"
    for rank, d in enumerate(results, start=1):
        if d[key] == answer:
            return rank
    return None


def evaluate(eval_set=EVAL_SET, methods=METHODS, k=5):
    """질문마다 각 방식에서 정답이 몇 위에 나왔는지 출력하고, 유형별·전체 recall@k와 MRR을 보여준다."""
    names = list(methods)
    print(f"정답의 순위 (상위 {k}개 안에 없으면 -)")
    print(f"  hybrid = 후보 {CANDIDATES}개, BM25 가중치 {W_BM25}, RRF 상수 {RRF_K}\n")
    print(f"{'유형':<9} {'질문':<36} {'정답':<10} " + " ".join(f"{n:>7}" for n in names))
    hits = {n: 0 for n in names}
    rr_sum = {n: 0.0 for n in names}   # 순위의 역수 합 (MRR 계산용)
    by_type = {}   # 유형별 집계: {"n": 질문 수, 방식 이름: 맞힌 수, ...}
    for qtype, query, answer in eval_set:
        row = by_type.setdefault(qtype, {"n": 0, **{n: 0 for n in names}})
        row["n"] += 1
        ranks = {}
        for name, fn in methods.items():
            r = rank_of(fn(query, k), answer)
            ranks[name] = r
            hits[name] += r is not None
            row[name] += r is not None
            rr_sum[name] += 1 / r if r else 0
        short = query if len(query) <= 20 else query[:19] + "…"
        cells = " ".join(f"{(str(ranks[n]) if ranks[n] else '-'):>7}" for n in names)
        print(f"{qtype:<9} {short:<36} {answer:<10} {cells}")

    print(f"\n유형별 recall@{k} (상위 {k}개 안에 정답이 든 질문 수)")
    for qtype, row in by_type.items():
        print(f"  {qtype:<9} " + ", ".join(f"{n} {row[n]}/{row['n']}" for n in names))
    total = len(eval_set)
    print(f"\n전체 recall@{k}: " + ", ".join(f"{n} {hits[n]}/{total}" for n in names))
    # MRR: 정답 순위의 역수 평균 (1위 1점, 2위 0.5점, 3위 0.33점 …, 못 찾으면 0점).
    #      recall이 같아도 정답을 얼마나 위에 올렸는지 비교할 수 있다.
    print("전체 MRR: " + ", ".join(f"{n} {rr_sum[n] / total:.3f}" for n in names))


# ──────────────────────────────────────────────────────────────
# RRF 조정 실험 (python search.py --sweep)
#   하이브리드의 세 가지 설정(rrf_k, candidates, w_bm25)을 바꿔 가며 같은 질문으로 비교한다.
#   조합마다 Elasticsearch를 다시 부르지 않도록, 질문마다 두 검색 결과를 한 번만 넉넉히(50개) 가져와 두고
#   계산(rrf)만 바꿔서 돌린다.
# ──────────────────────────────────────────────────────────────
SWEEP_RRF_K = (1, 10, 30, 60)
SWEEP_CANDIDATES = (5, 10, 20, 50)
SWEEP_W_BM25 = (1.0, 0.5, 0.3)


def sweep(k=5, max_candidates=50, top_n=10):
    print(f"질문 {len(EVAL_SET)}개의 BM25·kNN 결과를 {max_candidates}개씩 미리 가져오는 중...")
    cached = [(answer, bm25(query, max_candidates), knn(query, max_candidates))
              for _, query, answer in EVAL_SET]

    def measure(combine):
        """combine(bm25 결과, knn 결과) → 상위 k개. 질문 전체의 (recall, MRR)을 돌려준다."""
        ranks = [rank_of(combine(b, kn), answer) for answer, b, kn in cached]
        recall = sum(r is not None for r in ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        return recall, mrr

    total = len(EVAL_SET)
    print(f"\n기준선 (상위 {k}개)")
    for name, combine in (("bm25만", lambda b, kn: b[:k]), ("knn만", lambda b, kn: kn[:k])):
        recall, mrr = measure(combine)
        print(f"  {name:<8} recall {recall}/{total}  MRR {mrr:.3f}")

    rows = []
    for rrf_k in SWEEP_RRF_K:
        for cand in SWEEP_CANDIDATES:
            for w in SWEEP_W_BM25:
                recall, mrr = measure(lambda b, kn: rrf(b, kn, k, rrf_k, cand, w))
                rows.append((rrf_k, cand, w, recall, mrr))

    # recall이 높은 순, 같으면 MRR이 높은 순
    rows.sort(key=lambda r: (r[3], r[4]), reverse=True)
    current = next(r for r in rows if r[:3] == (RRF_K, CANDIDATES, W_BM25))
    print(f"\n하이브리드 조합 {len(rows)}개 중 상위 {top_n}개 (지금 설정: rrf_k={RRF_K}, candidates={CANDIDATES}, w_bm25={W_BM25})")
    print(f"  {'rrf_k':>5} {'후보 수':>6} {'w_bm25':>7} {'recall':>8} {'MRR':>7}")
    for r in rows[:top_n] + ([current] if current not in rows[:top_n] else []):
        mark = "  ← 지금 설정" if r is current else ""
        print(f"  {r[0]:>5} {r[1]:>6} {r[2]:>7} {r[3]:>5}/{total} {r[4]:>7.3f}{mark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 · kNN · 하이브리드 검색 비교")
    parser.add_argument("query", nargs="*", help="검색할 질문 (없으면 기본 질문)")
    parser.add_argument("--eval", action="store_true", help="준비한 질문으로 한 번에 비교")
    parser.add_argument("--sweep", action="store_true", help="RRF 설정 조합을 바꿔 가며 하이브리드 비교")
    parser.add_argument("--holdout", action="store_true", help="튜닝에 쓰지 않은 검증용 질문으로 비교")
    parser.add_argument("--index", default=INDEX, help=f"검색할 인덱스 (기본 {INDEX})")
    args = parser.parse_args()

    # 모듈 전역 INDEX를 바꾸면 bm25/knn이 이 인덱스를 검색한다.
    # (api.py는 search를 import해서 쓰므로 이 블록이 실행되지 않고, 항상 기본 인덱스를 쓴다)
    INDEX = args.index
    print(f"인덱스: {INDEX}")

    if args.sweep:
        sweep()
    elif args.holdout:
        evaluate(EVAL_SET_HOLDOUT)
    elif args.eval:
        evaluate()
    else:
        show(" ".join(args.query) or "입사한 지 1년 된 직원은 연차가 며칠이야?")
