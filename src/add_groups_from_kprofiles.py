"""나무위키 4세대 리스트에 있는 누락 그룹을 kprofiles에서 자동 스크랩.

흐름:
1) namu_4gen_groups.csv 로드 → members.csv에 없는 그룹만 남김
2) 각 그룹명을 여러 kprofiles slug 패턴으로 시도 (404 면 다음 패턴)
3) 페이지 발견 시:
   - Debut Date 파싱 → 오늘(기본 TODAY) 이후면 '미데뷔'로 skip
   - 해체/계약종료/유닛 키워드 보이면 skip
   - 멤버 리스트(영문 + 한국어 스테이지명) 추출
4) 정상 그룹이면 members.csv append (중복 member_id skip)

출력 로그: data/kprofiles_add_log.csv — 각 그룹별 상태 (added, skipped, not_found, not_debuted, disbanded, unit)
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
KPROFILES_BASE = "https://kprofiles.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
)

TODAY = date(2026, 4, 23)  # 오늘(호출 시 override 가능)

STAGE_NAME_WITH_KR_RE = re.compile(
    r"Stage\s*Name\s*:\s*(?:</span>\s*)*[\s\xa0]*"
    r"(?P<en>[A-Za-z][A-Za-z0-9 _\-\.']*?)\s*"
    r"\(\s*(?P<kr>[가-힣][가-힣 \-]*)\s*\)",
    re.IGNORECASE,
)
STAGE_NAME_ONLY_RE = re.compile(
    r"Stage\s*Name\s*:\s*(?:</span>\s*)*[\s\xa0]*(?P<en>[A-Za-z][A-Za-z0-9 _\-\.']*?)\s*[\(<]",
    re.IGNORECASE,
)
GROUP_KR_RE = re.compile(
    r"([A-Z][A-Za-z0-9!:'\.\s]{1,40})\s*\(\s*([가-힣][가-힣 \-]*)\s*\)\s*is\s+a",
    re.IGNORECASE,
)

# Debut Date 패턴들 — 여러 형식 모두 시도
DEBUT_PATTERNS = [
    re.compile(r"Debut\s*Date\s*:\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", re.IGNORECASE),
    re.compile(r"Debut\s*Date\s*:\s*(\d{4}[\./]\d{1,2}[\./]\d{1,2})", re.IGNORECASE),
    re.compile(r"Debut\s*Date\s*:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})", re.IGNORECASE),
    re.compile(r"debuted\s+on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})", re.IGNORECASE),
]
MONTH_EN = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12,
}

# 해체 표시
DISBAND_KEYWORDS = [
    re.compile(r"\bdisbanded\b", re.IGNORECASE),
    re.compile(r"\bcontract\s+ended\b", re.IGNORECASE),
    re.compile(r"\bcontract\s+terminated\b", re.IGNORECASE),
]
# 유닛 표시
UNIT_KEYWORDS = [
    re.compile(r"\bsub[-\s]?unit\s+of\b", re.IGNORECASE),
    re.compile(r"\bunit\s+group\s+of\b", re.IGNORECASE),
]
# 버츄얼 아이돌
VIRTUAL_KEYWORDS = [
    re.compile(r"\bvirtual\s+(girl|boy|K-?pop|idol)?\s*group\b", re.IGNORECASE),
    re.compile(r"\bvirtual\s+idol\s+group\b", re.IGNORECASE),
    re.compile(r"\bVTuber\b", re.IGNORECASE),
    re.compile(r"\bmetaverse\s+(girl|boy|idol)", re.IGNORECASE),
]
# 예능/프로젝트 그룹
PROJECT_KEYWORDS = [
    re.compile(r"\bproject\s+group\b", re.IGNORECASE),
    re.compile(r"\btemporary\s+group\b", re.IGNORECASE),
    re.compile(r"\bone[-\s]?time\s+project\b", re.IGNORECASE),
    re.compile(r"\bformed\s+through\s+.{0,40}\bvariety\s+show\b", re.IGNORECASE),
    re.compile(r"\bspecial\s+(project|collab)\b", re.IGNORECASE),
]

# 수동 블랙리스트 — kprofiles intro 문구로는 잡히지 않는 케이스
MANUAL_BLACKLIST: set[str] = {
    # 예능/프로젝트 그룹
    "환불원정대", "싹쓰리", "체크메이트", "음율", "재쓰비",
    "WSG워너비", "MSG워너비", "GOT the beat", "SUPER JUNIOR-L.S.S.",
    "Apink 초봄", "ARTBEAT v",
    # 버츄얼
    "이세계아이돌", "MAVE:", "PLAVE", "나빌레라", "ablume",
    # 솔로 아티스트 (그룹 X)
    "경서예지", "고막소년단", "신화 WDJ",
    # 기존 그룹의 유닛 (이름에 '/' 나 'X' 로 연결되지만 공백으로 끊어 써서 정규식 우회)
    "NCT 도재정", "정한X원우", "셔누 X 형원", "진진&라키", "문빈&산하",
}


def fetch_html(url: str, timeout: int = 15) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def slug_candidates(name: str) -> list[str]:
    """kprofiles 에서 시도할 slug 후보 (우선순위 순)."""
    # 소문자화, 특수문자→hyphen, 중복 hyphen 제거
    base = name.lower()
    base = re.sub(r"[&]", "and", base)
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    if not base:
        return []
    cand = []
    # 첫 버전: 그대로
    cand.append(f"{base}-members-profile")
    cand.append(f"{base}-profile")
    cand.append(f"{base}-members")
    cand.append(base)
    # prefix 특수 처리: 숫자-시작이나 symbol 포함 → 다른 형태 시도
    # KISS OF LIFE → kiss-of-life-members-profile (공백 → hyphen 자동 처리됨)
    # 특수: 이중 'ss' 케이스 (tripleS → tripless or triples)
    if name.lower().endswith("s") and len(name) > 4:
        alt = re.sub(r"s$", "ss", base) + "-members-profile"
        if alt not in cand:
            cand.append(alt)
    # 공백 제거 버전
    no_space = re.sub(r"[^a-z0-9]", "", name.lower())
    if no_space and no_space != base:
        cand.append(f"{no_space}-members-profile")
        cand.append(f"{no_space}-profile")
    return cand


def slug_for_id(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "unknown"


def parse_debut_date(html: str) -> date | None:
    for p in DEBUT_PATTERNS:
        m = p.search(html)
        if not m:
            continue
        s = m.group(1)
        # Try formats
        dp = _parse_date_str(s)
        if dp:
            return dp
    return None


def _parse_date_str(s: str) -> date | None:
    s = s.strip()
    # YYYY.MM.DD or YYYY/MM/DD
    m = re.match(r"(\d{4})[\./](\d{1,2})[\./](\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    # Month D, YYYY
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
    if m:
        mo = MONTH_EN.get(m.group(1).lower())
        if mo:
            try:
                return date(int(m.group(3)), mo, int(m.group(2)))
            except ValueError:
                return None
    # D Month YYYY
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        mo = MONTH_EN.get(m.group(2).lower())
        if mo:
            try:
                return date(int(m.group(3)), mo, int(m.group(1)))
            except ValueError:
                return None
    return None


def _head_text(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text[:4000]


def looks_disbanded(html: str) -> bool:
    head = _head_text(html)
    return any(p.search(head) for p in DISBAND_KEYWORDS)


def looks_like_unit(html: str) -> bool:
    head = _head_text(html)
    return any(p.search(head) for p in UNIT_KEYWORDS)


def looks_virtual(html: str) -> bool:
    head = _head_text(html)
    return any(p.search(head) for p in VIRTUAL_KEYWORDS)


def looks_project(html: str) -> bool:
    head = _head_text(html)
    return any(p.search(head) for p in PROJECT_KEYWORDS)


def extract_members(html: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in STAGE_NAME_WITH_KR_RE.finditer(html):
        en = m.group("en").strip()
        kr = m.group("kr").strip()
        if en in seen:
            continue
        seen.add(en)
        out.append((en, kr))
    if out:
        return out
    for m in STAGE_NAME_ONLY_RE.finditer(html):
        en = m.group("en").strip()
        if en in seen:
            continue
        seen.add(en)
        out.append((en, ""))
    return out


def extract_group_korean(html: str) -> str:
    m = GROUP_KR_RE.search(html)
    return m.group(2).strip() if m else ""


def load_existing_member_ids(members_csv: Path) -> set[str]:
    s: set[str] = set()
    if not members_csv.exists():
        return s
    with members_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            mid = (r.get("member_id") or "").strip()
            if mid:
                s.add(mid)
    return s


def load_existing_group_names(members_csv: Path) -> set[str]:
    s: set[str] = set()
    if not members_csv.exists():
        return s
    with members_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            g = (r.get("group_name") or "").strip()
            if g:
                s.add(g)
    return s


def load_namu_groups(namu_csv: Path) -> list[tuple[str, int, str]]:
    """(gender, debut_year, group_name) 리스트 반환."""
    out = []
    with namu_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append((r["gender"], int(r["debut_year"]), r["group_name"]))
    return out


def append_members(members_csv: Path, new_rows: list[dict[str, str]]) -> int:
    if not new_rows:
        return 0
    with members_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        fieldnames = ["member_id", "group_name", "member_name", "search_hint", "include_terms", "exclude_terms"]
    with members_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writerows(new_rows)
    return len(new_rows)


def find_kprofiles_page(group_name: str) -> tuple[str, str] | None:
    """→ (working_url, html) or None."""
    for slug in slug_candidates(group_name):
        url = f"{KPROFILES_BASE}/{slug}/"
        try:
            html = fetch_html(url)
            # 유효 페이지 확인: Stage Name 패턴 하나라도 있어야 진짜 멤버 프로필 페이지
            if "Stage Name" in html:
                return url, html
        except HTTPError as e:
            if e.code != 404:
                # 다른 에러면 약간 대기 후 다음 시도
                time.sleep(0.3)
            continue
        except (URLError, TimeoutError):
            continue
    return None


def process_group(
    group_name: str,
    debut_year: int,
    existing_group_names: set[str],
    existing_member_ids: set[str],
    today: date,
) -> tuple[str, list[dict[str, str]], str]:
    """→ (status, new_rows, note). status ∈ {added, skip_existing, not_found, not_debuted, disbanded, unit, virtual, project, blacklisted, no_members}."""
    if group_name in existing_group_names:
        return ("skip_existing", [], "already in members.csv")
    if group_name in MANUAL_BLACKLIST:
        return ("blacklisted", [], "on MANUAL_BLACKLIST (project/virtual/solo)")

    found = find_kprofiles_page(group_name)
    if found is None:
        return ("not_found", [], "no kprofiles page / slug unknown")
    url, html = found

    if looks_disbanded(html):
        return ("disbanded", [], "disbanded keyword in intro")
    if looks_virtual(html):
        return ("virtual", [], "virtual/VTuber keyword in intro")
    if looks_project(html):
        return ("project", [], "project/temporary group keyword in intro")
    if looks_like_unit(html):
        return ("unit", [], "sub-unit / unit keyword in intro")

    debut = parse_debut_date(html)
    if debut and debut > today:
        return ("not_debuted", [], f"debut date {debut} > today {today}")
    # 나무위키 연도가 미래인데 페이지에 debut 없으면 보수적 제외
    if not debut and debut_year > today.year:
        return ("not_debuted", [], f"namu debut_year {debut_year} > {today.year}, no page debut date")

    members = extract_members(html)
    if not members:
        return ("no_members", [], "no Stage Name patterns")

    group_kr = extract_group_korean(html)
    group_id = slug_for_id(group_name)

    new_rows: list[dict[str, str]] = []
    for en, kr in members:
        mid = f"{group_id}__{slug_for_id(en)}"
        if mid in existing_member_ids:
            continue
        existing_member_ids.add(mid)
        if group_kr and kr:
            hint = f'"{group_kr}" {kr}'
        elif group_kr:
            hint = f'"{group_kr}" {en}'
        elif kr:
            hint = f'"{group_name}" {kr}'
        else:
            hint = f'"{group_name}" {en}'
        new_rows.append({
            "member_id": mid,
            "group_name": group_name,
            "member_name": en,
            "search_hint": hint,
            "include_terms": "",
            "exclude_terms": "",
        })
    return ("added", new_rows, f"+{len(new_rows)} members (한글={group_kr or '-'}, debut={debut})")


def run(
    namu_csv: Path,
    members_csv: Path,
    log_csv: Path,
    today: date,
    delay_seconds: float,
    only_gender: str | None,
) -> None:
    namu = load_namu_groups(namu_csv)
    existing_names = load_existing_group_names(members_csv)
    existing_ids = load_existing_member_ids(members_csv)

    all_new_rows: list[dict[str, str]] = []
    log_rows: list[dict[str, str]] = []

    for gender, year, name in namu:
        if only_gender and gender != only_gender:
            continue
        status, new_rows, note = process_group(
            group_name=name,
            debut_year=year,
            existing_group_names=existing_names,
            existing_member_ids=existing_ids,
            today=today,
        )
        log_rows.append({
            "gender": gender,
            "debut_year": str(year),
            "group_name": name,
            "status": status,
            "note": note,
        })
        if status == "added":
            all_new_rows.extend(new_rows)
            existing_names.add(name)
        tag = {
            "added": "[+]", "skip_existing": "[=]", "not_found": "[x]",
            "not_debuted": "[⏳]", "disbanded": "[💀]", "unit": "[U]", "no_members": "[?]",
        }.get(status, "")
        print(f"  {tag} {gender} {year}  {name:25s} — {note}")
        if status == "added" and delay_seconds > 0:
            time.sleep(delay_seconds)

    added_count = append_members(members_csv, all_new_rows)
    print()
    # Summary
    from collections import Counter
    status_counts = Counter(r["status"] for r in log_rows)
    print("=== Summary ===")
    for s, c in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")
    print(f"Total new members appended: {added_count}")

    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gender", "debut_year", "group_name", "status", "note"])
        w.writeheader()
        w.writerows(log_rows)
    print(f"Log: {log_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-scrape kprofiles for groups in namu_4gen_groups.csv.")
    parser.add_argument("--namu", default="data/namu_4gen_groups.csv")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--log", default="data/kprofiles_add_log.csv")
    parser.add_argument("--delay-seconds", type=float, default=0.4)
    parser.add_argument("--today", default=TODAY.isoformat(), help="YYYY-MM-DD, 기본은 오늘.")
    parser.add_argument("--gender", choices=["M", "F"], default=None, help="특정 성별만 처리.")
    args = parser.parse_args()

    today = date.fromisoformat(args.today)
    run(
        namu_csv=ROOT_DIR / args.namu,
        members_csv=ROOT_DIR / args.members,
        log_csv=ROOT_DIR / args.log,
        today=today,
        delay_seconds=args.delay_seconds,
        only_gender=args.gender,
    )


if __name__ == "__main__":
    main()
