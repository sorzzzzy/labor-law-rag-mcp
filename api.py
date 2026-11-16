"""검색 API: 질문을 받아 근로기준법 조항과 출처를 JSON으로 돌려준다.

3단계에서 만들 Kotlin MCP 서버가 이 API를 호출한다.
검색과 답변 생성을 분리해 두면, 나중에 Claude API로 답변을 만들 때도 이 API를 그대로 쓸 수 있다.

실행: uvicorn api:app --port 8000
문서: http://localhost:8000/docs  (브라우저에서 바로 호출해볼 수 있는 화면)
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query

import search  # search.py의 bm25 / knn / hybrid를 그대로 재사용


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 뜰 때 임베딩 모델을 미리 불러둔다. 첫 요청이 모델 로딩 때문에 느려지지 않게.
    search.model()
    yield


app = FastAPI(title="근로기준법 검색 API", lifespan=lifespan)


def source_label(d):
    """답변에 붙일 출처 문구. 예: 근로기준법 제60조 ①항 (시행 2026. 10. 8., 법률 제21533호)"""
    para = f" {d['paragraph']}항" if d.get("paragraph") else ""
    return f"{d['law']} {d['article']}{para} ({d['version']})"


@app.get("/search")
def search_law(
    q: str = Query(..., description="검색할 질문이나 키워드"),
    # 기본값을 knn으로 둔 이유: 5단계 평가(질문 8개)에서 knn이 8/8로 가장 좋았다
    mode: Literal["knn", "bm25", "hybrid"] = Query("knn", description="검색 방식"),
    k: int = Query(5, ge=1, le=20, description="돌려받을 조항 수"),
):
    results = search.METHODS[mode](q, k)
    return {
        "query": q,
        "mode": mode,
        "results": [
            {
                "rank": rank,
                "id": d["id"],
                "article": d["article"],
                "paragraph": d.get("paragraph"),
                "title": d.get("title"),
                "chapter": d.get("chapter"),
                "text": d["text"],               # 근거로 보여줄 조문 원문
                "source": source_label(d),       # 답변에 붙일 출처
                "score": round(d["score"], 4),
            }
            for rank, d in enumerate(results, start=1)
        ],
    }


@app.get("/health")
def health():
    """Elasticsearch 연결과 색인된 문서 수 확인용."""
    return {
        "elasticsearch": search.es.info()["version"]["number"],
        "index": search.INDEX,
        "docs": search.es.count(index=search.INDEX)["count"],
    }
