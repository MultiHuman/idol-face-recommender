"""kprofiles.com 그룹 페이지에서 "Boy Group" / "Girl Group" 텍스트를 뽑아서
data/group_genders.csv 를 만든다. build_member_vectors 가 이걸 우선 적용해서
genderage 모델의 개별 오판정을 교정한다.

페이지에는 보통 "XYZ is a South Korean boy group ..." 같은 문장이 있다.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
KPROFILES_BASE = "https://kprofiles.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
)

# _SLUG_OVERRIDES 를 fetch_anchors 와 동일하게 재사용
from src.fetch_anchors import _SLUG_OVERRIDES, _URL_OVERRIDES, _group_to_slug


# kprofiles 페이지가 404 / ambiguous 로 나오는 그룹의 성별을 수동 확정.
# fetch 전에 체크해서 네트워크 요청 없이 바로 결과 반환.
MANUAL_OVERRIDES: dict[str, str] = {
    "&TEAM": "M",
    "2F": "M",
    "CLASS:y": "F",
    "Dragon Pony": "M",
    "Hi-Fi Un!corn": "M",
    "JWiiver": "M",
    "PRIKIL": "F",
    "SEVENUS": "M",
    "THE SSYNDROME": "M",
    "VAYONN": "M",
    "VCHA": "F",
    "VI'ENX": "M",
    "We;Na": "F",
    "XG": "F",
    "XLOV": "M",
    # timeout 이나 slug 이상으로 fetch 실패한 유명 그룹 — 일반 상식 기반
    "STAYC": "F",
    "SUPER JUNIOR": "M",
    "SEVENTEEN": "M",
    "MONSTA X": "M",
    "BTOB": "M",
    "NCT": "M",
    "BTS": "M",
    "TXT": "M",
    "TWICE": "F",
    "Stray Kids": "M",
    "Red Velvet": "F",
    "ITZY": "F",
    "IU": "F",
    "INFINITE": "M",
}


# 페이지 본문 (breadcrumb/JSON-LD)에 articleSection: ["Kpop Boy Groups"|"Kpop Girl Groups"] 가
# 단 한 번 명확히 나온다. 사이드바 메뉴의 "Boy Groups"/"Girl Groups" 는 양쪽 다 등장하므로
# 단순 단어 카운트로는 판정 불가. articleSection 을 직접 매칭한다.
GIRL_PATTERNS = [
    re.compile(r'articleSection[^\]]{0,40}Kpop\s+Girl\s+Groups', re.IGNORECASE),
    re.compile(r'is\s+a\s+[^<]{0,80}?\bgirl\s+group\b', re.IGNORECASE),
]
BOY_PATTERNS = [
    re.compile(r'articleSection[^\]]{0,40}Kpop\s+Boy\s+Groups', re.IGNORECASE),
    re.compile(r'is\s+a\s+[^<]{0,80}?\bboy\s+group\b', re.IGNORECASE),
]


def fetch_html(url: str, timeout: int = 15) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def classify_gender(html: str) -> str:
    # raw HTML 에 patterns 을 적용 — articleSection 은 JSON-LD script 안이라 tag-stripped 본문에도 보이지만
    # 확실한 앵커를 유지하려고 raw 그대로 씀.
    girl_hit = sum(1 for p in GIRL_PATTERNS if p.search(html))
    boy_hit = sum(1 for p in BOY_PATTERNS if p.search(html))
    if girl_hit > boy_hit:
        return "F"
    if boy_hit > girl_hit:
        return "M"
    return ""


def load_groups(members_csv: Path) -> list[str]:
    groups: set[str] = set()
    with members_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            g = (r.get("group_name") or "").strip()
            if g:
                groups.add(g)
    return sorted(groups)


def load_existing(output_csv: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not output_csv.exists():
        return result
    with output_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            g = (r.get("group_name") or "").strip()
            gender = (r.get("gender") or "").strip()
            if g and gender:
                result[g] = gender
    return result


def run(members_csv: Path, output_csv: Path, delay_ms: int, overwrite: bool) -> None:
    groups = load_groups(members_csv)
    existing = {} if overwrite else load_existing(output_csv)
    results: list[tuple[str, str, str]] = []

    ok, fail = 0, 0
    for group_name in groups:
        if group_name in existing:
            results.append((group_name, existing[group_name], "cached"))
            ok += 1
            continue

        if group_name in MANUAL_OVERRIDES:
            g = MANUAL_OVERRIDES[group_name]
            results.append((group_name, g, "manual_override"))
            print(f"[man]  {group_name:30s} → {g} (manual)")
            ok += 1
            continue

        slug_raw = _group_to_slug(group_name)
        slug = _SLUG_OVERRIDES.get(slug_raw, slug_raw)
        url_override = _URL_OVERRIDES.get(slug_raw)
        url = (
            f"{KPROFILES_BASE}/{url_override}/" if url_override
            else f"{KPROFILES_BASE}/{slug}-members-profile/"
        )
        try:
            html = fetch_html(url)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[fail] {group_name}: {exc}")
            results.append((group_name, "", f"fetch_fail: {exc}"))
            fail += 1
            time.sleep(delay_ms / 1000)
            continue

        gender = classify_gender(html)
        if gender:
            ok += 1
            print(f"[ok]   {group_name:30s} → {gender}")
        else:
            fail += 1
            print(f"[amb]  {group_name:30s} → (ambiguous)")
        results.append((group_name, gender, "downloaded" if gender else "ambiguous"))
        time.sleep(delay_ms / 1000)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group_name", "gender", "status"])
        writer.writeheader()
        for g, gender, status in results:
            writer.writerow({"group_name": g, "gender": gender, "status": status})
    print(f"\nWrote {len(results)} group genders to {output_csv}. OK={ok} FAIL={fail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer group gender from kprofiles pages.")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--output", default="data/group_genders.csv")
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        members_csv=ROOT_DIR / args.members,
        output_csv=ROOT_DIR / args.output,
        delay_ms=args.delay_ms,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
