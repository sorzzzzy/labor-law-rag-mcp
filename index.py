"""청크를 임베딩(벡터)으로 바꾸고, 원문·벡터·출처를 Elasticsearch 인덱스 하나에 저장한다.

흐름: chunks.jsonl 읽기 → 인덱스 새로 만들기(매핑) → bge-m3로 임베딩 → 일괄 색인(bulk)

사용법: python index.py data/chunks.jsonl
"""
import json
import sys

import torch
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

ES_URL = "http://localhost:9200"
INDEX = "labor_law"
MODEL_NAME = "BAAI/bge-m3"   # 한국어를 지원하는 다국어 임베딩 모델, 1024차원
DIMS = 1024

# ──────────────────────────────────────────────────────────────
# 인덱스 설정과 매핑 (RDB의 테이블 스키마에 해당)
# ──────────────────────────────────────────────────────────────
SETTINGS = {
    # 노트북 한 대에서 도는 실습용이라 샤드 1개, 복제본 0개
    "number_of_shards": 1,
    "number_of_replicas": 0,
}

MAPPINGS = {
    "properties": {
        # keyword: 정확히 일치하는 값으로 찾거나 거르는 필드 (분석하지 않고 통째로 저장)
        "id": {"type": "keyword"},          # 예: "제60조①"
        "law": {"type": "keyword"},
        "version": {"type": "keyword"},
        "chapter": {"type": "keyword"},
        "article": {"type": "keyword"},     # 예: "제60조" → "제60조" 그대로 필터 가능
        "paragraph": {"type": "keyword"},   # 예: "①"
        "sort_key": {"type": "integer"},

        # text + nori: 형태소로 나눠 색인하는 필드 → BM25 키워드 검색 대상
        "title": {"type": "text", "analyzer": "nori"},
        "content": {"type": "text", "analyzer": "nori"},   # 상위 제목 + 원문

        # 원문은 답변에 근거로 보여주기만 하고 검색에는 쓰지 않는다 (색인 안 함, 저장만)
        "text": {"type": "text", "index": False},

        # dense_vector: 임베딩을 저장하고 kNN(벡터) 검색에 쓰는 필드
        "embedding": {
            "type": "dense_vector",
            "dims": DIMS,
            "index": True,               # kNN 검색용 인덱스(HNSW 그래프)를 만든다
            "similarity": "cosine",      # 가까움을 코사인 유사도로 잰다
            # 기본값은 벡터를 압축(양자화)해서 저장하는 방식이다.
            # 실습에서는 원래 값 그대로 비교하도록 압축 없는 HNSW를 쓴다.
            "index_options": {"type": "hnsw"},
        },
    }
}


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def create_index(es):
    """같은 이름의 인덱스가 있으면 지우고 새로 만든다 (실습이라 매번 깨끗하게)."""
    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
    es.indices.create(index=INDEX, settings=SETTINGS, mappings=MAPPINGS)


def embed(chunks):
    """청크의 content를 bge-m3로 벡터로 바꾼다."""
    # M1의 GPU(MPS)를 쓸 수 있으면 쓰고, 안 되면 CPU로
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"임베딩 모델 불러오는 중... ({MODEL_NAME}, {device})")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = [c["content"] for c in chunks]
    # normalize_embeddings=True: 벡터 길이를 1로 맞춘다. 코사인 유사도 계산이 안정적이 된다.
    vectors = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    return vectors.tolist()


def index_chunks(es, chunks, vectors):
    """청크와 벡터를 묶어 bulk API로 한 번에 색인한다."""
    actions = (
        {
            "_index": INDEX,
            "_id": c["id"],                        # 같은 id면 덮어쓴다 → 다시 실행해도 중복이 안 생김
            "_source": {**c, "embedding": vec},
        }
        for c, vec in zip(chunks, vectors)
    )
    ok, errors = helpers.bulk(es, actions, raise_on_error=False)
    es.indices.refresh(index=INDEX)                # 방금 넣은 문서가 바로 검색되도록
    return ok, errors


def main(path):
    es = Elasticsearch(ES_URL)
    print(f"Elasticsearch {es.info()['version']['number']} 연결")

    chunks = load_chunks(path)
    print(f"청크 {len(chunks)}개 읽음")

    create_index(es)
    print(f"인덱스 '{INDEX}' 생성")

    vectors = embed(chunks)
    print(f"임베딩 완료: {len(vectors)}개 x {len(vectors[0])}차원")

    ok, errors = index_chunks(es, chunks, vectors)
    print(f"색인 성공 {ok}개, 실패 {len(errors)}개")

    count = es.count(index=INDEX)["count"]
    print(f"인덱스 문서 수: {count} (청크 수 {len(chunks)}와 같아야 정상)")


if __name__ == "__main__":
    main(sys.argv[1])
