"""법제처에서 받은 근로기준법 HWP(XML 형식) 파일에서 본문 텍스트만 뽑아 저장한다.

사용법: python prepare_data.py data/labor_law.hwp data/labor_law.txt
"""
import html
import re
import sys

src, out = sys.argv[1], sys.argv[2]
xml = open(src, encoding="utf-8").read()

# 1) 본문(<BODY>)의 문단(<P>)마다 글자(<CHAR>)를 이어 붙여 한 줄로 만든다
body = xml[xml.find("<BODY"):]
lines = []
for p in re.findall(r"<P\b[^>]*>(.*?)</P>", body, flags=re.S):
    chars = re.findall(r"<CHAR[^>]*>(.*?)</CHAR>", p, flags=re.S)
    text = html.unescape("".join(chars)).replace(" ", " ")
    lines.append(re.sub(r"<[^>]+>", "", text).rstrip())

# 2) "제1장"부터 "부칙" 바로 앞까지만 남긴다 (머리말·담당부서 연락처·부칙 제외)
start = next(i for i, l in enumerate(lines) if re.match(r"\s*제1장\s", l))
end = next(i for i, l in enumerate(lines) if re.match(r"\s*부칙", l))
lines = [l for l in lines[start:end] if l.strip()]

open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")

chapters = sum(bool(re.match(r"\s*제\d+장", l)) for l in lines)
articles = sum(bool(re.match(r"제\d+조", l)) for l in lines)
print(f"저장: {out}  줄 {len(lines)}  장 {chapters}  조 {articles}")
