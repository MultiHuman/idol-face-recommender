"""나무위키 '4세대 아이돌' 문서에서 보이/걸그룹 전체 목록을 추출.

본문에 '[ 전체 목록 ] 구분 데뷔 연도 팀명 보이그룹 YYYY group1 , group2 ...
걸그룹 YYYY ... 혼성 ... 남성 ... 여성 ...' 구조로 정돈된 표가 있다.

해체/계약종료/프로젝트종료 마커가 붙은 그룹은 제외.
그룹명에 '&'(유닛 콜라보), 숫자만 있는 그룹 (NCT 도재정 등 유닛 프로젝트) 도 제외.

출력: data/namu_4gen_groups.csv — columns: gender (F/M), debut_year, group_name
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
URL = "https://namu.wiki/w/4%EC%84%B8%EB%8C%80%20%EC%95%84%EC%9D%B4%EB%8F%8C"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
)

# 해체/종료 마커 — 이게 그룹명 뒤 [...] 형태로 붙어있으면 제외
DISBAND_MARKERS = (
    "해체", "계약종료", "사실상해체", "프로젝트종료",
)

# 유닛/프로젝트/비정상 케이스 — 그룹명 기반 제외 패턴
SKIP_PATTERNS = [
    re.compile(r"&"),  # 멤버 콜라보 (문빈&산하, 셔누 X 형원 등)
    re.compile(r"\s+[Xx×]\s+"),  # ditto
    re.compile(r"/"),  # NCT 도재정 같은 유닛은 NCT/...로 표기 드물지만 방어
    re.compile(r"워너비$"),  # WSG워너비, MSG워너비 등 프로젝트 그룹
    re.compile(r"^음율$|^재쓰비$|^싹쓰리$"),  # 혼성 프로젝트 하드 제외
    re.compile(r"\d+기$"),  # 가비엔제이/5기
]


def fetch_html() -> str:
    req = Request(URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


def strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_4gen_section(text: str) -> str:
    """5번째 '[ 전체 목록 ]' 이 4세대. 시작 위치 ~ 다음 '[편집]' 섹션 전까지 반환."""
    positions = [m.start() for m in re.finditer(r"\[\s*전체\s*목록\s*\]", text)]
    if len(positions) < 5:
        # fallback: the one containing '2024' and '2025'
        for p in positions:
            snippet = text[p : p + 4000]
            if "2024" in snippet and "2025" in snippet:
                return snippet
        raise RuntimeError(f"전체 목록 section not located (found {len(positions)})")
    start = positions[4]
    # 다음 '[편집]' 마커까지 또는 4000자
    end = text.find("[편집]", start + 50)
    if end < 0 or end - start > 6000:
        end = start + 6000
    return text[start:end]


def extract_group_list(section: str, label: str, next_label: str | None) -> list[tuple[int, str]]:
    """section 내에서 `label`과 `next_label` 사이 텍스트를 추출하고 (year, group) 페어 리스트 반환."""
    start = section.find(label)
    if start < 0:
        return []
    start += len(label)
    if next_label:
        end = section.find(next_label, start)
        if end < 0:
            end = len(section)
    else:
        end = len(section)
    body = section[start:end]

    # 연도-그룹 추출: "YYYY group1 , group2 , ... YYYY group ..."
    # year pattern: \b(20\d\d)\b
    tokens = re.split(r"\s(?=(?:20\d\d)\s)", body)
    # 위 split 은 lookahead 라 년도 직전에만 잘림. 각 토큰은 "YYYY group1 , group2 , ..." 형식
    results: list[tuple[int, str]] = []
    for tok in tokens:
        tok = tok.strip()
        m = re.match(r"(20\d\d)\s+(.+)$", tok)
        if not m:
            continue
        year = int(m.group(1))
        groups_str = m.group(2)
        # Split by comma
        for name in groups_str.split(","):
            name = name.strip()
            if not name:
                continue
            # 대괄호로 둘러싼 마커 제거 후 본체 판정
            core = re.sub(r"\[[^\]]*\]", "", name).strip()
            if not core:
                continue
            # 해체 마커 검사
            markers = re.findall(r"\[([^\]]+)\]", name)
            if any(any(dm in m for dm in DISBAND_MARKERS) for m in markers):
                continue
            # skip patterns
            if any(p.search(core) for p in SKIP_PATTERNS):
                continue
            # Filter: group name must start with alphanumeric or hangul
            if len(core) < 2:
                continue
            results.append((year, core))
    return results


def parse_4gen_groups(html: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    text = strip_html(html)
    section = find_4gen_section(text)
    boys = extract_group_list(section, "보이그룹", "걸그룹")
    girls = extract_group_list(section, "걸그룹", "혼성")
    return boys, girls


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse 4th-gen idol group list from namu.wiki.")
    parser.add_argument("--output", default="data/namu_4gen_groups.csv")
    args = parser.parse_args()

    html = fetch_html()
    boys, girls = parse_4gen_groups(html)

    out = ROOT_DIR / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gender", "debut_year", "group_name"])
        w.writeheader()
        for year, name in boys:
            w.writerow({"gender": "M", "debut_year": year, "group_name": name})
        for year, name in girls:
            w.writerow({"gender": "F", "debut_year": year, "group_name": name})

    print(f"Wrote {len(boys)} boy groups + {len(girls)} girl groups → {out}")
    print()
    print("Boy groups:")
    for year, name in boys:
        print(f"  {year}  {name}")
    print()
    print("Girl groups:")
    for year, name in girls:
        print(f"  {year}  {name}")


if __name__ == "__main__":
    main()
