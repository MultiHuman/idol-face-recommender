"""
Fetch stage names from kprofiles.com and compare/update members.csv.

Usage:
    python -m src.sync_member_names --report          # show mismatches only
    python -m src.sync_member_names --apply           # update members.csv in-place
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .crawl import USER_AGENT, load_members, MemberRecord

KPROFILES_BASE = "https://kprofiles.com"
STAGE_NAME_RE = re.compile(
    r"Stage Name\s*:\s*([^\n(<]+?)(?:\s*\(([^)]*)\))?\s*(?=<|\n|$)",
    re.UNICODE,
)


# --------------------------------------------------------------------------- #
# Group name → kprofiles URL slug
# --------------------------------------------------------------------------- #

_SLUG_OVERRIDES: dict[str, str] = {
    "&team": "and-team",
    "nine.i": "nine-i",
    "e'last": "elast",
    "monsta x": "monsta-x",
    "omega x": "omega-x",
    "one pact": "one-pact",
    "the kingdom": "the-kingdom",
    "the wind": "the-wind",
    "xdinary heroes": "xdinary-heroes",
    "le sserafim": "le-sserafim",
    "nct wish": "nct-wish",
}


def group_to_slug(group_name: str) -> str:
    key = group_name.lower().strip()
    if key in _SLUG_OVERRIDES:
        return _SLUG_OVERRIDES[key]
    slug = re.sub(r"['\.\u2019]", "", key)
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


# --------------------------------------------------------------------------- #
# Fetch & parse
# --------------------------------------------------------------------------- #

_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]+")
_JAPANESE_RE = re.compile(r"[\u3040-\u30FF\u31F0-\u31FF]")


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return html_lib.unescape(text)


def _extract_korean_name(raw: str) -> str:
    """
    From a stage name that may contain Japanese/slash notation like
    'ケイ / 케이' or '의주 / ウィジュ', return only the Korean part.
    """
    # If contains slash, pick the segment with the most Hangul
    if "/" in raw:
        parts = [p.strip() for p in raw.split("/")]
        raw = max(parts, key=lambda p: len(_HANGUL_RE.findall(p)))
    # Strip any remaining Japanese characters
    raw = _JAPANESE_RE.sub("", raw).strip()
    return raw


_GROUP_KO_RE = re.compile(
    r"\(\s*Korean\s*:\s*([\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F][^)]*?)\s*[);]",
    re.UNICODE,
)
_GROUP_HANGUL_RE = re.compile(r"^[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]+$")


def fetch_group_info(group_slug: str, timeout: int = 15) -> tuple[str, list[tuple[str, str]]]:
    """Return (group_korean_name, [(en_stage_name, ko_stage_name), ...]) for a group."""
    url = f"{KPROFILES_BASE}/{group_slug}-members-profile/"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {url}")
    except URLError as exc:
        raise RuntimeError(f"Network error: {url} — {exc}")

    plain = _strip_html(raw)

    # 1) 그룹 한국어명 추출 — "(Korean: 튜넥스;" 같은 패턴
    group_ko = ""
    m = _GROUP_KO_RE.search(plain)
    if m:
        candidate = m.group(1).strip().split(";")[0].strip()
        # 첫 한글 토큰만 (조사/공백 제거)
        tokens = candidate.split()
        if tokens and _GROUP_HANGUL_RE.match(tokens[0]):
            group_ko = tokens[0]

    # 2) 멤버 stage name 추출
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in STAGE_NAME_RE.finditer(plain):
        en_raw = m.group(1).strip()
        ko_raw = (m.group(2) or "").strip()

        # Some groups have 'Japanese / Korean' or 'DK / Dokyeom' in the main name field
        if _JAPANESE_RE.search(en_raw):
            en = _extract_korean_name(en_raw)
        elif "/" in en_raw:
            # Take the first (shorter) part as the stage name
            en = en_raw.split("/")[0].strip()
        else:
            en = en_raw
        # en should be a name — skip if empty or suspiciously long
        if not en or len(en) > 40:
            continue
        if en in seen:
            continue

        ko = _extract_korean_name(ko_raw) if ko_raw else ""
        seen.add(en)
        results.append((en, ko))
    return group_ko, results


def fetch_stage_names(group_slug: str, timeout: int = 15) -> list[tuple[str, str]]:
    """Backwards-compat wrapper that returns only the member list."""
    return fetch_group_info(group_slug, timeout=timeout)[1]


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #

def _normalize(name: str) -> str:
    return re.sub(r"[\s\-\.]", "", name).lower()


def _match_member(
    csv_member: MemberRecord,
    kprofiles_names: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Find the best kprofiles entry for a CSV member."""
    target = _normalize(csv_member.member_name)

    # 1. Exact match
    for en, ko in kprofiles_names:
        if _normalize(en) == target:
            return en, ko

    # 2. kprofiles name may include surname: 'Ahn Yujin' → last token 'Yujin'
    for en, ko in kprofiles_names:
        last_token = _normalize(en.split()[-1]) if en.split() else ""
        if last_token and last_token == target:
            return en, ko

    # 3. CSV name may include surname: last token of CSV matches full kprofiles name
    target_last = _normalize(csv_member.member_name.split()[-1]) if csv_member.member_name.split() else target
    for en, ko in kprofiles_names:
        if _normalize(en) == target_last:
            return en, ko

    # 4. Substring
    for en, ko in kprofiles_names:
        ne = _normalize(en)
        if target in ne or ne in target:
            return en, ko

    return None


# --------------------------------------------------------------------------- #
# search_hint update: replace the Korean name part
# --------------------------------------------------------------------------- #

def _update_search_hint(hint: str, old_ko: str, new_ko: str) -> str:
    """Replace the first occurrence of old_ko in hint with new_ko."""
    if not old_ko or not new_ko or old_ko == new_ko:
        return hint
    return hint.replace(old_ko, new_ko, 1)


def _extract_ko_name_from_hint(hint: str, group_name_ko: str | None = None) -> str:
    """
    Best-effort: extract the member Korean name from a search_hint like
    '에스파 카리나 얼굴' → 'カリナ' not needed; returns 'カリナ'.
    We just strip the trailing '얼굴' and the leading group portion.
    """
    hint = hint.strip()
    if hint.endswith("얼굴"):
        hint = hint[:-2].strip()
    # The last whitespace-separated token is the member name
    parts = hint.split()
    return parts[-1] if parts else ""


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def run(members_csv: Path, apply: bool, delay_ms: int) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    with members_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Group members by group_name
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        gname = row.get("group_name", "").strip()
        groups.setdefault(gname, []).append(row)

    changes: list[dict[str, str]] = []

    for group_name, group_rows in groups.items():
        slug = group_to_slug(group_name)
        try:
            group_ko, kp_names = fetch_group_info(slug)
        except RuntimeError as exc:
            print(f"[skip] {group_name} ({slug}): {exc}", file=sys.stderr)
            continue

        if not kp_names:
            print(f"[warn] {group_name}: no stage names parsed", file=sys.stderr)
            continue

        print(
            f"[ok]   {group_name}"
            f"{' (Korean: ' + group_ko + ')' if group_ko else ''}"
            f": {len(kp_names)} members from kprofiles"
        )

        for row in group_rows:
            member_id = row.get("member_id", "")
            current_en = row.get("member_name", "").strip()
            current_hint = row.get("search_hint", "").strip()

            # Build a fake MemberRecord just for matching
            mr = MemberRecord(
                member_id=member_id,
                group_name=group_name,
                member_name=current_en,
                search_hint=current_hint,
                include_terms=(),
                exclude_terms=(),
            )
            match = _match_member(mr, kp_names)
            if match is None:
                print(f"  [no match] {member_id} ('{current_en}')", file=sys.stderr)
                continue

            new_en, new_ko = match
            current_ko = _extract_ko_name_from_hint(current_hint)
            new_hint = _update_search_hint(current_hint, current_ko, new_ko) if new_ko else current_hint

            # 그룹 한국어명을 search_hint에 반영 (영문 그룹명을 한국어로 교체)
            if group_ko and group_name in new_hint:
                new_hint = new_hint.replace(group_name, group_ko)

            en_changed = _normalize(new_en) != _normalize(current_en)
            hint_changed = new_hint != current_hint

            if en_changed or hint_changed:
                changes.append({
                    "member_id": member_id,
                    "old_member_name": current_en,
                    "new_member_name": new_en if en_changed else current_en,
                    "old_search_hint": current_hint,
                    "new_search_hint": new_hint,
                })
                print(
                    f"  [diff] {member_id}: "
                    + (f"name '{current_en}' → '{new_en}'" if en_changed else "")
                    + (" | " if en_changed and hint_changed else "")
                    + (f"hint '{current_hint}' → '{new_hint}'" if hint_changed else "")
                )
                if apply:
                    row["member_name"] = new_en if en_changed else current_en
                    row["search_hint"] = new_hint

        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    print(f"\n총 {len(changes)}건 변경{'(적용됨)' if apply else '(--apply로 적용 가능)'}.")

    if apply and changes:
        with members_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"저장 완료: {members_csv}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync member names in members.csv from kprofiles.com."
    )
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update members.csv (default: report only).",
    )
    parser.add_argument("--delay-ms", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        members_csv=Path(args.members),
        apply=args.apply,
        delay_ms=args.delay_ms,
    )


if __name__ == "__main__":
    main()
