# 근로기준법 RAG 검색 + MCP 실습

근로기준법을 조·항 단위로 검색해 근거 조항과 출처를 돌려주는 검색 파이프라인을 직접 만들고, 이를 MCP 도구로 Claude에 연결해 근거 기반으로 답하게 하는 실습입니다.

> 진행 중인 실습입니다. 현재 RAG 검색까지 완료했고, MCP 연결을 진행할 예정입니다.

## 구조

```
Claude 데스크톱 앱 (답변 생성)
   │  stdio · JSON-RPC
   ▼
Kotlin MCP 서버 ── search_labor_law 도구        ← 진행 예정
   │  HTTP
   ▼
검색 API (Python, FastAPI) ── Elasticsearch 9.3.8 (nori + dense_vector)
```

## 진행 상황

- [x] Elasticsearch + nori 형태소 분석기 실행 환경 (Docker)
- [x] 근로기준법 원문 텍스트 추출 (`prepare_data.py`)
- [x] 조·항 단위 청킹, 상위 제목과 출처 메타데이터 (`chunk.py`)
- [x] bge-m3 임베딩과 색인 (`index.py`)
- [x] BM25 · kNN · 하이브리드(RRF 직접 구현) 검색 비교 (`search.py`)
- [x] 검색 API (`api.py`)
- [ ] Kotlin MCP 서버와 Claude 데스크톱 앱 연결

## 기술 스택

- Elasticsearch 9.3.8, analysis-nori (Docker)
- Python 3.12, sentence-transformers, 임베딩 모델 BAAI/bge-m3 (1024차원)
- FastAPI, uvicorn
- (예정) Kotlin, MCP Kotlin SDK

## 설계에서 정한 것

- **청킹:** 조 단위를 기본으로 하고, 500자를 넘는 조는 항 단위로 나눴습니다. 모든 청크 앞에 `근로기준법 > 제4장 근로시간과 휴식 > 제60조(연차 유급휴가)` 같은 상위 제목을 붙여, 항만 떼어낸 청크도 맥락을 잃지 않게 했습니다.
- **출처:** 청크마다 법 이름, 시행 버전, 장, 조, 항을 저장해 답변에 근거 조항을 표시할 수 있게 했습니다.
- **매핑:** 본문은 nori로 분석하는 text 필드(BM25용), 임베딩은 dense_vector(1024차원, 코사인 유사도) 필드에 함께 저장했습니다.
- **하이브리드:** BM25와 kNN의 점수 단위가 달라, 순위 기반으로 합치는 RRF를 직접 구현했습니다.

## 검색 비교 결과

법률 용어를 그대로 쓴 질문 4개와, 같은 내용을 일상 표현으로 풀어 쓴 질문 4개로 비교했습니다. 정답 조문이 상위 5개 안에 들어온 질문 수(recall@5)입니다.

| 방식 | 법률 용어 질문 | 일상 표현 질문 | 전체 |
|---|---|---|---|
| BM25 | 4/4 | 1/4 | 5/8 |
| kNN | 4/4 | 4/4 | 8/8 |
| 하이브리드 (RRF) | 4/4 | 3/4 | 7/8 |

- 일상 표현 질문("회사가 갑자기 내일부터 나오지 말라고 하면…")은 법률 용어("해고의 예고")와 겹치는 단어가 없어 BM25가 찾지 못했습니다.
- 하이브리드가 kNN보다 낮았습니다. RRF에서는 두 검색의 후보에 모두 걸친 문서가 한쪽에서만 1위인 문서보다 항상 점수가 높아, BM25가 빗나간 질문에서는 kNN의 정답이 밀려났습니다. 그래서 검색 API의 기본 방식은 kNN으로 정했습니다.

## 실행 방법

```bash
# 1. Elasticsearch + nori
docker build -t es-nori:9.3.8 es/
docker run -d --name es-rag -p 9200:9200 \
  -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" es-nori:9.3.8

# 2. Python 환경
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. 데이터 준비와 색인 (data/labor_law.txt가 저장소에 포함되어 있어 chunk.py부터 실행해도 됩니다)
python chunk.py data/labor_law.txt data/chunks.jsonl
python index.py data/chunks.jsonl

# 4. 검색 비교
python search.py --eval

# 5. 검색 API
uvicorn api:app --port 8000
# http://localhost:8000/docs
```

## 데이터 출처

- 근로기준법 (시행 2026. 10. 8., 법률 제21533호), 국가법령정보센터
- 원본 HWP 파일은 저장소에 포함하지 않았습니다. 국가법령정보센터에서 내려받아 `python prepare_data.py data/labor_law.hwp data/labor_law.txt`로 변환할 수 있습니다.
