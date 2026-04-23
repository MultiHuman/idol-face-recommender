"""Scrape kprofiles.com to build a verified members.csv."""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Group name -> kprofiles URL slug
# We'll scrape each page and extract members
GROUPS = {
    # Girl groups
    "aespa": "aespa-members-profile",
    "IVE": "ive-members-profile",
    "LE SSERAFIM": "le-sserafim-members-profile",
    "NewJeans": "newjeans-members-profile",
    "NMIXX": "nmixx-members-profile",
    # 2020 boy groups
    "ENHYPEN": "enhypen-members-profile",
    "P1Harmony": "p1harmony-members-profile",
    "TREASURE": "treasure-members-profile",
    "CRAVITY": "cravity-members-profile",
    "DKB": "dkb-members-profile",
    "E'LAST": "elast-members-profile",
    "BAE173": "bae173-members-profile",
    "DRIPPIN": "drippin-members-profile",
    "GHOST9": "ghost9-members-profile",
    "LUCY": "lucy-band-members-profile",
    "MCND": "mcnd-members-profile",
    "WEi": "wei-members-profile",
    # 2021
    "BLITZERS": "blitzers-members-profile",
    "EPEX": "epex-members-profile",
    "OMEGA X": "omega-x-members-profile",
    "The KingDom": "the-kingdom-members-profile",
    "Xdinary Heroes": "xdinary-heroes-members-profile",
    "NTX": "ntx-members-profile",
    # 2022
    "&TEAM": "and-team-members-profile",
    "NINE.i": "nine-i-members-profile",
    "TAN": "tan-members-profile",
    "TEMPEST": "tempest-members-profile",
    "TNX": "tnx-members-profile",
    "TRENDZ": "trendz-members-profile",
    "YOUNITE": "younite-members-profile",
    # 2023
    "PLAVE": "plave-members-profile",
    "ZEROBASEONE": "zerobaseone-members-profile",
    "EVNNE": "evnne-members-profile",
    "HORI7ON": "hori7on-members-profile",
    "LUN8": "lun8-members-profile",
    "ONE PACT": "one-pact-members-profile",
    "TIOT": "tiot-members-profile",
    "WHIB": "whib-members-profile",
    "XODIAC": "xodiac-members-profile",
    "xikers": "xikers-members-profile",
    "n.SSign": "nssign-members-profile",
    "8TURN": "8turn-members-profile",
    "82MAJOR": "82major-members-profile",
    "CMDM": "cmdm-command-the-m-boys-members-profile",
    "AMPERS&ONE": "ampersone-members-profile",
    "POW": "pow-members-profile",
    "FANTASY BOYS": "fantasy-boys-members-profile",
    "Hi-Fi Un!corn": "hi-fi-uncorn-members-profile",
    "SEVENUS": "sevenus-members-profile",
    "The Wind": "the-wind-members-profile",
    # 2024
    "NCT": "nct-members-profile",
    "NCT WISH": "nct-wish-members-profile",
    "TWS": "tws-members-profile",
    "SEVENTEEN": "seventeen-members-profile",
    "SUPER JUNIOR": "super-junior-members-profile",
    "MONSTA X": "monsta-x-members-profile",
    "BTOB": "btob-members-profile",
    "RIIZE": "riize-members-profile",
    "BOYNEXTDOOR": "boynextdoor-members-profile",
    "NEXZ": "nexz-members-profile",
    "DXMON": "dxmon-members-profile",
    "NCHIVE": "nchive-members-profile",
    "ALL(H)OURS": "allhours-members-profile",
    "ARrC": "arrc-members-profile",
    "DAYCHILD": "daychild-members-profile",
    "Dragon Pony": "dragon-pony-members-profile",
    "NOWZ": "nowadays-members-profile",
    "WAKER": "waker-members-profile",
    "IDID": "idid-members-profile",
    "KickFlip": "kickflip-members-profile",
    # 2025
    "CORTIS": "cortis-members-profile",
    "AHOF": "ahof-members-profile",
    "AxMxP": "axmxp-members-profile",
    "BE BOYS": "be-boys-members-profile",
    "CLOSE YOUR EYES": "close-your-eyes-members-profile",
    "idntt": "idntt-members-profile",
    "NEWBEAT": "newbeat-members-profile",
    "NouerA": "nouera-members-profile",
    "POLARIX": "polarix-members-profile",
    "XLOV": "xlov-members-profile",
    # 2026
    "ALPHA DRIVE ONE": "alpha-drive-one-ald1-members-profile",
    "ADAP": "adap-members-profile",
    "AND2BLE": "and2ble-members-profile",
    "DAILY:DIRECTION": "dailydirection-members-profile",
    "hrtz.wav": "hrtz-wav-members-profile",
    "KEYVITUP": "keyvitup-members-profile",
    "VAYONN": "vay-onn-members-profile",
    "LNGSHOT": "lngshot-members-profile",
    "MODYSSEY": "modyssey-members-profile",
    "NAZE": "c9rookies-members-profile",
    "NXD": "nxd-members-profile",
    "TUNEXX": "tunexx-members-profile",
    "YUHZ": "yuhz-members-profile",
    "THE SSYNDROME": "the-ssyndrome-members-profile",
}

# Korean group name mapping for search_hint
GROUP_KR = {
    "aespa": "에스파",
    "IVE": "아이브",
    "LE SSERAFIM": "르세라핌",
    "NewJeans": "뉴진스",
    "NMIXX": "엔믹스",
    "ENHYPEN": "엔하이픈",
    "P1Harmony": "피원하모니",
    "TREASURE": "트레저",
    "CRAVITY": "크래비티",
    "DKB": "다크비",
    "E'LAST": "엘라스트",
    "BAE173": "BAE173",
    "DRIPPIN": "드리핀",
    "GHOST9": "고스트나인",
    "LUCY": "루시",
    "MCND": "엠씨엔디",
    "WEi": "위아이",
    "BLITZERS": "블리처스",
    "EPEX": "이펙스",
    "OMEGA X": "오메가엑스",
    "The KingDom": "더킹덤",
    "Xdinary Heroes": "엑스디너리히어로즈",
    "NTX": "엔티엑스",
    "&TEAM": "앤팀",
    "NINE.i": "나인아이",
    "TAN": "TAN",
    "TEMPEST": "템페스트",
    "TNX": "티엔엑스",
    "TRENDZ": "트렌드지",
    "YOUNITE": "유나이트",
    "PLAVE": "플레이브",
    "ZEROBASEONE": "제로베이스원",
    "EVNNE": "이븐",
    "HORI7ON": "호라이즌",
    "LUN8": "루네이트",
    "ONE PACT": "원팩트",
    "TIOT": "티아이오티",
    "WHIB": "휘브",
    "XODIAC": "소디엑",
    "xikers": "싸이커스",
    "n.SSign": "엔싸인",
    "8TURN": "에잇턴",
    "82MAJOR": "팔이메이저",
    "CMDM": "CMDM",
    "AMPERS&ONE": "앰퍼샌드원",
    "POW": "POW",
    "FANTASY BOYS": "판타지보이즈",
    "Hi-Fi Un!corn": "하이파이유니콘",
    "SEVENUS": "세븐어스",
    "The Wind": "더윈드",
    "NCT": "NCT",
    "NCT WISH": "엔시티위시",
    "TWS": "투어스",
    "SEVENTEEN": "세븐틴",
    "SUPER JUNIOR": "슈퍼주니어",
    "MONSTA X": "몬스타엑스",
    "BTOB": "비투비",
    "RIIZE": "라이즈",
    "BOYNEXTDOOR": "보이넥스트도어",
    "NEXZ": "넥스지",
    "DXMON": "다이몬",
    "NCHIVE": "엔카이브",
    "ALL(H)OURS": "올아워즈",
    "ARrC": "ARrC",
    "DAYCHILD": "데이차일드",
    "Dragon Pony": "드래곤포니",
    "NOWZ": "나우즈",
    "WAKER": "WAKER",
    "IDID": "아이디드",
    "KickFlip": "킥플립",
    "CORTIS": "코르티스",
    "AHOF": "AHOF",
    "AxMxP": "AxMxP",
    "BE BOYS": "비보이즈",
    "CLOSE YOUR EYES": "클로즈유어아이즈",
    "idntt": "idntt",
    "NEWBEAT": "뉴비트",
    "NouerA": "누에라",
    "POLARIX": "폴라릭스",
    "XLOV": "XLOV",
    "ALPHA DRIVE ONE": "알파드라이브원",
    "ADAP": "ADAP",
    "AND2BLE": "앤더블",
    "DAILY:DIRECTION": "데일리디렉션",
    "hrtz.wav": "hrtz.wav",
    "KEYVITUP": "KEYVITUP",
    "VAYONN": "베이온",
    "LNGSHOT": "LNGSHOT",
    "MODYSSEY": "MODYSSEY",
    "NAZE": "NAZE",
    "NXD": "NXD",
    "TUNEXX": "TUNEXX",
    "YUHZ": "YUHZ",
    "THE SSYNDROME": "더씬드롬",
}


def make_member_id(group_name: str, member_name: str) -> str:
    """Generate a member_id like 'aespa__karina'."""
    g = re.sub(r"[^a-zA-Z0-9]", "_", group_name.lower()).strip("_")
    g = re.sub(r"_+", "_", g)
    m = re.sub(r"[^a-zA-Z0-9]", "_", member_name.lower()).strip("_")
    m = re.sub(r"_+", "_", m)
    return f"{g}__{m}"


def extract_members(page, url: str) -> list[dict]:
    """Extract current members from a kprofiles page."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2000)
    except Exception as e:
        print(f"  [error] Failed to load {url}: {e}")
        return []

    # Extract the page text content
    content = page.content()

    # Strategy: find all <p> tags that contain "Stage Name:" or "Birth Name:" patterns
    # kprofiles uses <p> tags with <strong> for member info
    members = []

    # Extract full text blocks from entry-content
    try:
        raw = page.evaluate(r"""
        () => {
            const results = [];
            const paragraphs = document.querySelectorAll('.entry-content p, .entry-content div');
            let currentMember = null;
            let hitFormer = false;

            for (const p of paragraphs) {
                const text = p.textContent.trim();
                if (!text) continue;

                const lowerText = text.toLowerCase();

                // Stop at former members section
                if ((lowerText.includes('former member') || lowerText.includes('has departed') ||
                     lowerText.includes('left the group') || lowerText.includes('disbanded') ||
                     lowerText.includes('hiatus')) &&
                    !lowerText.includes('stage name')) {
                    if (currentMember && currentMember.stageName) {
                        results.push(currentMember);
                        currentMember = null;
                    }
                    hitFormer = true;
                }
                if (hitFormer) continue;

                // Match "Stage Name:" or "Stage / Birth Name:" patterns
                const stageMatch = text.match(/Stage\s*(?:\/\s*Birth)?\s*Name\s*:\s*(.+?)(?:\n|$)/i);
                if (stageMatch) {
                    if (currentMember && currentMember.stageName) {
                        results.push(currentMember);
                    }
                    currentMember = {stageName: '', koreanName: '', birthName: ''};
                    const fullLine = stageMatch[1].trim();
                    // Extract Korean in parentheses (various bracket styles)
                    const korMatch = fullLine.match(/[\(\（]([가-힣\s]+)[\)\）]/);
                    const nameOnly = fullLine.replace(/[\(\（].*?[\)\）]/g, '').replace(/\u00a0/g, ' ').trim();
                    // Clean up trailing junk
                    currentMember.stageName = nameOnly.split('\n')[0].trim();
                    if (korMatch) {
                        currentMember.koreanName = korMatch[1].trim();
                    }
                }

                if (!currentMember) continue;

                // Korean Name line
                const korNameMatch = text.match(/Korean\s*Name\s*:\s*([가-힣\s]+)/i);
                if (korNameMatch && !currentMember.koreanName) {
                    currentMember.koreanName = korNameMatch[1].trim();
                }

                // Birth Name line - extract Korean from parentheses or direct Korean text
                const birthMatch = text.match(/Birth\s*Name\s*:\s*(.+?)(?:\n|$)/i);
                if (birthMatch && !currentMember.koreanName) {
                    const birthLine = birthMatch[1].trim();
                    const korBirth = birthLine.match(/[\(\（]([가-힣\s]+)[\)\）]/);
                    if (korBirth) {
                        currentMember.koreanName = korBirth[1].trim();
                    } else {
                        const pureKor = birthLine.match(/([가-힣]{2,})/);
                        if (pureKor) {
                            currentMember.koreanName = pureKor[1].trim();
                        }
                    }
                }
            }

            if (currentMember && currentMember.stageName) {
                results.push(currentMember);
            }

            return results;
        }
        """)
    except Exception as e:
        print(f"  [error] JS extraction failed: {e}")
        return []

    return raw


def main():
    output_path = Path("data/members_verified.csv")
    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, channel="msedge")
        page = browser.new_page()

        for group_name, slug in GROUPS.items():
            url = f"https://kprofiles.com/{slug}/"
            print(f"[scrape] {group_name}: {url}")

            members = extract_members(page, url)
            kr_group = GROUP_KR.get(group_name, group_name)

            if not members:
                print(f"  [warn] No members found for {group_name}")
                continue

            for m in members:
                stage = m.get("stageName", "").strip()
                korean = m.get("koreanName", "").strip()
                if not stage:
                    continue

                member_id = make_member_id(group_name, stage)
                search_hint = f"{kr_group} {korean} 얼굴" if korean else f"{kr_group} {stage} 얼굴"

                results.append({
                    "member_id": member_id,
                    "group_name": group_name,
                    "member_name": stage,
                    "search_hint": search_hint,
                    "include_terms": "",
                    "exclude_terms": "",
                })
                print(f"  {stage} ({korean})")

            time.sleep(0.5)

        browser.close()

    # Write CSV
    fieldnames = ["member_id", "group_name", "member_name", "search_hint", "include_terms", "exclude_terms"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Wrote {len(results)} members to {output_path}")


if __name__ == "__main__":
    main()
