"""kprofiles 에서 각 멤버의 공식 프로필 사진을 가져와 .cache/anchors/ 에 저장.

매핑 전략:
1) 그룹 HTML 페이지 다운로드
2) 'Stage Name:</span> NAME (한국어)' 패턴으로 모든 stage name 위치 추출
3) 각 stage name 위치에서 backward 로 가장 가까운 <img src=".../uploads/..."> 찾음
4) 그 이미지가 해당 stage name 의 프로필 사진 → 다운로드

members.csv 의 member_name 과 kprofiles 의 stage name 을 fuzzy matching 해서
member_id ↔ image 매핑을 만든다.
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

STAGE_NAME_RE = re.compile(
    r'Stage Name\s*:\s*(?:</span>\s*)+[\s\xa0]*([A-Za-z][A-Za-z0-9 _\-\.]*?)\s*[\(<]',
    re.IGNORECASE,
)
IMG_RE = re.compile(
    r'<img[^>]*src=["\'](https?://[^"\']+uploads[^"\']+\.(?:jpg|jpeg|png|webp))["\'][^>]*>',
    re.IGNORECASE,
)

# 멤버 프로필이 아닌 로고/배너 이미지 제외
_BLOCKED_SUBSTRINGS = ("herald_logo", "-logo", "_logo", "logo.jpg", "logo.png",
                       "banner", "footer", "header", "icon", "avatar")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _group_to_slug(group_name: str) -> str:
    """members.csv 의 group_name → kprofiles slug."""
    slug = group_name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


# 특수 slug override: kprofiles 의 실제 URL 경로
_SLUG_OVERRIDES = {
    "aespa": "aespa",
    "ive": "ive",
    "newjeans": "newjeans",
    "le-sserafim": "le-sserafim",
    "nct": "nct-u",
    "p1harmony": "p1harmony",
    "and-team": "and-team",
    "nine-i": "nine-i",
    "hori7on": "hori7on",
    "zerobaseone": "zerobaseone",
    "xdinary-heroes": "xdinary-heroes",
    "the-kingdom": "the-kingdom",
    "eighty-two-major": "82major",
    "nct-wish": "nct-wish",
    "bae173": "bae173",
    "boynextdoor": "boynextdoor",
    "ghost9": "ghost9",
    "tws": "tws",
    "idntt": "idntt",
    "ntx": "ntx",
    # 신규 매핑 (2026-04-08)
    "alpha-drive-one": "alpha-drive-one-ald1",
    "hi-fi-un-corn": "hi-fi-uncorn",
    "hifi-unicorn": "hi-fi-uncorn",
    "lngshot": "more-vision-boys",
    "naze": "c9rookies",
    "cmdm": "cmdm-command-the-m-boys",
    "nowz": "nowadays",
    "lucy": "lucy-band",
    "n-ssign": "nssign",
    "ampers-one": "ampersone",
    "ampersone": "ampersone",
    "all-h-ours": "all-hours",
    "allhours": "all-hours",
    "daily-direction": "dailydirection",
    "just-b": "just-b",
    "justb": "just-b",
    "wei": "wei",
    "e-last": "elast",
    "elast": "elast",
}

# 특수 URL 패턴 override: 기본은 {slug}-members-profile/ 인데 예외 그룹
_URL_OVERRIDES = {
    "nmixx": "nmixx-profile",
}


def fetch_html(url: str, timeout: int = 15) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_bytes(url: str, timeout: int = 15) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Referer": KPROFILES_BASE})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _is_blocked_image(url: str) -> bool:
    low = url.lower()
    return any(sub in low for sub in _BLOCKED_SUBSTRINGS)


def extract_stage_to_image(html: str) -> list[tuple[str, str]]:
    """HTML → [(stage_name, image_url), ...]"""
    img_positions = [
        (m.start(), m.end(), m.group(1))
        for m in IMG_RE.finditer(html)
        if not _is_blocked_image(m.group(1))
    ]
    results: list[tuple[str, str]] = []
    seen_stage: set[str] = set()
    for m in STAGE_NAME_RE.finditer(html):
        stage = m.group(1).strip()
        if stage in seen_stage:
            continue
        seen_stage.add(stage)
        # 가장 가까운 이전 img
        prev_img: str | None = None
        for _, end, src in img_positions:
            if end < m.start():
                prev_img = src
            else:
                break
        if prev_img:
            results.append((stage, prev_img))
    return results


def match_member_to_stage(
    en_name: str,
    stage_list: list[tuple[str, str]],
) -> str | None:
    """멤버 영문명 → stage name matching (fuzzy: normalized 비교)"""
    target = _normalize(en_name)
    if not target:
        return None

    # exact normalized match
    for stage, img in stage_list:
        if _normalize(stage) == target:
            return img
    # prefix match (DK vs Dokyeom, JM vs Jiminie 같은 경우)
    for stage, img in stage_list:
        ns = _normalize(stage)
        if ns.startswith(target) or target.startswith(ns):
            return img
    return None


def load_members(members_csv: Path) -> list[dict[str, str]]:
    with members_csv.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_members(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        g = row.get("group_name", "").strip()
        groups.setdefault(g, []).append(row)
    return groups


def run(
    members_csv: Path,
    output_dir: Path,
    log_csv: Path,
    delay_ms: int,
    overwrite: bool,
) -> None:
    rows = load_members(members_csv)
    groups = group_members(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict[str, str]] = []
    total_ok = 0
    total_fail = 0

    for group_name, group_rows in sorted(groups.items()):
        slug_raw = _group_to_slug(group_name)
        slug = _SLUG_OVERRIDES.get(slug_raw, slug_raw)
        url_override = _URL_OVERRIDES.get(slug_raw)
        if url_override:
            url = f"{KPROFILES_BASE}/{url_override}/"
        else:
            url = f"{KPROFILES_BASE}/{slug}-members-profile/"

        try:
            html = fetch_html(url)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[fail] {group_name} ({slug}): {exc}")
            for row in group_rows:
                log_rows.append({
                    "member_id": row["member_id"],
                    "anchor_path": "",
                    "source_url": "",
                    "status": "group_page_fail",
                    "error": str(exc),
                })
                total_fail += 1
            continue

        stage_list = extract_stage_to_image(html)
        if not stage_list:
            print(f"[warn] {group_name}: no stage names parsed")

        for row in group_rows:
            mid = row["member_id"]
            en = row.get("member_name", "").strip()
            dst = output_dir / f"{mid}.jpg"

            if dst.exists() and not overwrite:
                log_rows.append({
                    "member_id": mid, "anchor_path": str(dst.relative_to(ROOT_DIR)),
                    "source_url": "", "status": "cached", "error": "",
                })
                total_ok += 1
                continue

            img_url = match_member_to_stage(en, stage_list)
            if img_url is None:
                print(f"  [no match] {mid} ('{en}')")
                log_rows.append({
                    "member_id": mid, "anchor_path": "",
                    "source_url": "", "status": "no_match", "error": "",
                })
                total_fail += 1
                continue

            try:
                img_bytes = fetch_bytes(img_url)
                dst.write_bytes(img_bytes)
                print(f"  [ok] {mid} <- {img_url[-60:]}")
                log_rows.append({
                    "member_id": mid, "anchor_path": str(dst.relative_to(ROOT_DIR)),
                    "source_url": img_url, "status": "downloaded", "error": "",
                })
                total_ok += 1
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                print(f"  [fail] {mid}: {exc}")
                log_rows.append({
                    "member_id": mid, "anchor_path": "",
                    "source_url": img_url, "status": "download_fail", "error": str(exc),
                })
                total_fail += 1

        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["member_id", "anchor_path", "source_url", "status", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(log_rows)

    print(f"\n총 {total_ok}명 앵커 확보, {total_fail}명 실패. 로그: {log_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch kprofiles anchor images for each member.")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--output-dir", default=".cache/anchors")
    parser.add_argument("--log", default="data/anchor_log.csv")
    parser.add_argument("--delay-ms", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        members_csv=ROOT_DIR / args.members,
        output_dir=ROOT_DIR / args.output_dir,
        log_csv=ROOT_DIR / args.log,
        delay_ms=args.delay_ms,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
