"""근로기준법 텍스트를 검색 단위(청크)로 나눠 JSONL로 저장한다.

규칙
- 기본 단위는 "조". 조 하나가 주제 하나를 다뤄서 의미가 잘 안 끊긴다.
- 조가 MAX_CHARS보다 길면 "항" 단위로 나눈다. 관련 없는 항까지 딸려오는 걸 막는다.
- 모든 청크 앞에 "근로기준법 > 장 > 조(제목)"을 붙인다(문맥 증강).
- 삭제된 조·항, [본조신설 ...] 같은 개정 메모는 뺀다.

사용법: python chunk.py data/labor_law.txt data/chunks.jsonl
"""
import json  # 청크를 JSON 문자열로 저장할 때 사용 (Kotlin의 Jackson/kotlinx.serialization 역할)
import re    # 정규식 (Kotlin의 Regex)
import sys   # 실행할 때 넘긴 인자(sys.argv)를 읽을 때 사용

# ──────────────────────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────────────────────

# 모든 청크에 붙일 출처 정보
#   답변에 "근거: 근로기준법(시행 ...) 제60조 ①항"처럼 표시할 때 쓴다.
LAW = "근로기준법"
VERSION = "시행 2026. 10. 8., 법률 제21533호"

# 이 글자 수보다 긴 조는 항 단위로 나눠짐
#   500자면 대부분의 조는 통째로 들어가고, 제2조(정의)·제60조(연차)처럼 항이 많은 긴 조만 나뉜다.
MAX_CHARS = 500

# ──────────────────────────────────────────────────────────────
# 줄 종류를 구분하는 정규식
#   ^ : 줄의 시작,  $ : 줄의 끝,  \s* : 공백 0개 이상,  \d+ : 숫자 1개 이상
#   ( ... )  : 꺼내 쓸 그룹 (m.group(1), m.group(2) ...)
#   (?: ...) : 묶기만 하고 꺼내지는 않는 그룹,  ? : 있어도 되고 없어도 됨
# ──────────────────────────────────────────────────────────────

# 장
#   예: "       제4장 근로시간과 휴식", "       제6장의2 직장 내 괴롭힘의 금지"
#   group(1) = "제4장" / "제6장의2",  group(2) = "근로시간과 휴식"
RE_CHAPTER = re.compile(r"^\s*(제\d+장(?:의\d+)?)\s+(.+)$")

# 조
#   예: "제60조(연차 유급휴가) ① 사용자는 ...", "제76조의2(직장 내 괴롭힘의 금지) ...", "제35조 삭제"
#   group(1) = "제60조" (조 번호 전체)    group(2) = "60" (본 번호)
#   group(3) = "2" ("의2"의 번호, 없으면 None)
#   group(4) = "연차 유급휴가" (괄호 안 제목, 없으면 None)
#   group(5) = 제목 뒤에 오는 나머지 본문 ("① 사용자는 ...")
RE_ARTICLE = re.compile(r"^(제(\d+)조(?:의(\d+))?)(?:\(([^)]*)\))?\s*(.*)$")

# 항
#   예: "  ② 사용자는 ..."  → group(1) = "②",  group(2) = "사용자는 ..."
#   [①-⑳] 은 원문자 ①부터 ⑳까지 중 한 글자
RE_PARAGRAPH = re.compile(r"^\s*([①-⑳])\s*(.*)$")

# 메모
#   예: "  [본조신설 2024. 10. 22.]", "  [시행일: 2027. 1. 1.] ..." 처럼 대괄호로 시작하는 줄
#   조문 내용이 아니라 개정 이력이라 검색에 노이즈만 되므로 버린다.
RE_NOTE = re.compile(r"^\s*\[.*\]")


def parse(lines):
    """줄 단위 텍스트를 [{장, 조, 제목, 항 목록}] 구조로 만든다.

    한 줄씩 읽으면서 "지금 어느 장, 어느 조 안에 있는지"를 기억해 두고,
    줄 종류(장/조/항/호)에 따라 알맞은 자리에 넣는다.
    """
    articles = []    # 결과: 조 목록
    chapter = None   # 지금 읽고 있는 장 이름 (예: "제4장 근로시간과 휴식")
    cur = None       # 지금 채우고 있는 조

    for raw in lines:
        line = raw.rstrip()  # 줄 끝 공백 제거

        # 빈 줄이나 개정 메모는 건너뛴다
        if not line.strip() or RE_NOTE.match(line):
            continue

        # 장 줄이면 현재 장 이름만 바꾸고 다음 줄로
        # (:= 는 "대입하면서 조건으로도 쓰기". Kotlin의 RE.find(line)?.let { m -> ... } 와 같은 역할)
        if m := RE_CHAPTER.match(line):
            chapter = f"{m.group(1)} {m.group(2).strip()}"
            continue

        # 조 줄이면 새 조를 시작한다
        if m := RE_ARTICLE.match(line):
            no, main, sub, title, rest = m.groups()
            cur = {
                "chapter": chapter,
                "article": no,
                # 정렬용 숫자: 제60조 → 6000, 제76조의2 → 7602
                "sort_key": int(main) * 100 + int(sub or 0),
                "title": title,
                "intro": [],         # 항 없이 바로 시작하는 본문 (예: 제1조처럼 항이 없는 조)
                "paragraphs": [],    # [(①, [줄...]), (②, [줄...]), ...]
                # "제35조 삭제"처럼 제목 없이 "삭제"만 있으면 삭제된 조
                "deleted": rest.strip().startswith("삭제") and not title,
            }
            articles.append(cur)
            # 같은 줄에 붙어 있는 나머지 본문("① 사용자는 ...")을 아래에서 이어서 처리한다
            line = rest
            if not line.strip():
                continue

        # 첫 조가 나오기 전의 줄은 무시
        if cur is None:
            continue

        if m := RE_PARAGRAPH.match(line):
            # 항 줄: 새 항을 시작한다
            cur["paragraphs"].append((m.group(1), [m.group(2).strip()]))
        elif cur["paragraphs"]:
            # 항이 아닌 줄(호 "1. 2. ..." 등)은 직전 항에 붙인다.
            # 호만 따로 떼면 무엇에 대한 목록인지 알 수 없기 때문.
            cur["paragraphs"][-1][1].append(line.strip())
        else:
            # 아직 항이 없으면 조 본문(intro)으로
            cur["intro"].append(line.strip())

    # 삭제된 조는 빼고 돌려준다 (Kotlin의 articles.filter { !it.deleted })
    return [a for a in articles if not a["deleted"]]


def header(a):
    """청크 앞에 붙일 상위 제목 줄을 만든다.

    예: "근로기준법 > 제4장 근로시간과 휴식 > 제60조(연차 유급휴가)"
    항만 떼어낸 청크도 무엇에 대한 내용인지 알 수 있게 해준다(문맥 증강).
    """
    head = f"{a['article']}({a['title']})" if a["title"] else a["article"]
    return f"{LAW} > {a['chapter']} > {head}"


def to_chunks(a):
    """조 하나를 청크 하나, 또는 항별 여러 청크로 만든다.

    yield 는 결과를 하나씩 내보내는 문법 (Kotlin의 sequence { yield(...) } 와 같음).
    """
    # "③ 삭제"처럼 삭제된 항은 뺀다
    paras = [(mark, "\n".join(ls)) for mark, ls in a["paragraphs"] if " ".join(ls).strip() != "삭제"]

    # 조 전체 본문 = (항 없이 시작하는 본문) + (각 항)
    body_parts = (["\n".join(a["intro"])] if a["intro"] else []) + [f"{mark} {t}" for mark, t in paras]
    body = "\n".join(body_parts)

    # 모든 청크에 공통으로 들어갈 출처 메타데이터
    base = {
        "law": LAW, "version": VERSION, "chapter": a["chapter"],
        "article": a["article"], "title": a["title"], "sort_key": a["sort_key"],
    }

    # 청크의 두 가지 텍스트
    #   text    : 조문 원문만. 답변에 근거 원문을 보여줄 때 쓴다.
    #   content : 상위 제목 줄 + 원문. 검색(BM25)과 임베딩에 쓴다.
    # {**base, ...} 는 base를 복사하면서 필드를 더하는 문법 (Kotlin data class의 copy()와 비슷)

    # 짧은 조, 또는 항이 없는 조는 조 전체를 청크 하나로
    if len(body) <= MAX_CHARS or not paras:
        yield {**base, "id": a["article"], "paragraph": None,
               "text": body, "content": f"{header(a)}\n{body}"}
        return

    # 긴 조는 항 단위로 나눈다. id는 "제60조①"처럼 조 번호 + 항 번호
    for mark, t in paras:
        text = f"{mark} {t}"
        yield {**base, "id": f"{a['article']}{mark}", "paragraph": mark,
               "text": text, "content": f"{header(a)}\n{text}"}


def main(src, out):
    # 1) 텍스트 파일을 줄 단위로 읽는다
    lines = open(src, encoding="utf-8").read().splitlines()

    # 2) 구조로 만들고 → 3) 청크로 나눈다 (Kotlin의 articles.flatMap { toChunks(it) })
    articles = parse(lines)
    chunks = [c for a in articles for c in to_chunks(a)]

    # 4) JSONL로 저장: 한 줄에 청크 하나씩 JSON
    #    ensure_ascii=False 로 해야 한글이 \uXXXX 가 아니라 그대로 저장된다
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 5) 통계 출력: 청크가 너무 길거나 짧지 않은지 확인용
    lengths = sorted(len(c["content"]) for c in chunks)
    split_articles = {c["article"] for c in chunks if c["paragraph"]}  # 항 단위로 나뉜 조 (중복 없는 set)
    print(f"조 {len(articles)}개 -> 청크 {len(chunks)}개")
    print(f"항 단위로 나눈 조: {len(split_articles)}개")
    print(f"청크 길이(글자): 최소 {lengths[0]}, 중간 {lengths[len(lengths)//2]}, 최대 {lengths[-1]}")
    longest = max(chunks, key=lambda c: len(c["content"]))
    print(f"가장 긴 청크: {longest['id']} ({len(longest['content'])}자)")


# 이 파일을 직접 실행했을 때만 main()을 부른다 (다른 파일에서 import할 때는 실행 안 됨)
# sys.argv[1] = 입력 파일 경로, sys.argv[2] = 출력 파일 경로
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
