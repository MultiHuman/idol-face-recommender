"""Append non-disbanded missing 2012-2017 K-pop groups to members.csv.

This is intentionally one-shot and idempotent: existing member_id/group gender
entries are skipped, so it is safe to rerun after partial changes.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MEMBERS_CSV = ROOT_DIR / "data" / "members.csv"
GROUP_GENDERS_CSV = ROOT_DIR / "data" / "group_genders.csv"

# (member_id, group_name, member_name, search_hint, include_terms, exclude_terms)
NEW_ROWS: list[tuple[str, str, str, str, str, str]] = [
    # VIXX
    ("vixx__n", "VIXX", "N", '"빅스" 엔', "VIXX|빅스", ""),
    ("vixx__leo", "VIXX", "Leo", '"빅스" 레오', "VIXX|빅스", ""),
    ("vixx__ken", "VIXX", "Ken", '"빅스" 켄', "VIXX|빅스", ""),
    ("vixx__hyuk", "VIXX", "Hyuk", '"빅스" 혁', "VIXX|빅스", ""),

    # DAY6
    ("day6__sungjin", "DAY6", "Sungjin", '"데이식스" 성진', "DAY6|데이식스", ""),
    ("day6__young_k", "DAY6", "Young K", '"데이식스" 영케이', "DAY6|데이식스", ""),
    ("day6__wonpil", "DAY6", "Wonpil", '"데이식스" 원필', "DAY6|데이식스", ""),
    ("day6__dowoon", "DAY6", "Dowoon", '"데이식스" 도운', "DAY6|데이식스", ""),

    # N.Flying
    ("n_flying__seunghyub", "N.Flying", "Seunghyub", '"엔플라잉" 이승협', "N.Flying|엔플라잉", ""),
    ("n_flying__hun", "N.Flying", "Hun", '"엔플라잉" 차훈', "N.Flying|엔플라잉", ""),
    ("n_flying__jaehyun", "N.Flying", "Jaehyun", '"엔플라잉" 김재현', "N.Flying|엔플라잉", ""),
    ("n_flying__hweseung", "N.Flying", "Hweseung", '"엔플라잉" 유회승', "N.Flying|엔플라잉", ""),
    ("n_flying__dongsung", "N.Flying", "Dongsung", '"엔플라잉" 서동성', "N.Flying|엔플라잉", ""),

    # EXID
    ("exid__solji", "EXID", "Solji", '"이엑스아이디" 솔지', "EXID|이엑스아이디", ""),
    ("exid__elly", "EXID", "Elly", '"이엑스아이디" 엘리', "EXID|이엑스아이디", ""),
    ("exid__hani", "EXID", "Hani", '"이엑스아이디" 하니', "EXID|이엑스아이디", ""),
    ("exid__hyelin", "EXID", "Hyelin", '"이엑스아이디" 혜린', "EXID|이엑스아이디", ""),
    ("exid__jeonghwa", "EXID", "Jeonghwa", '"이엑스아이디" 정화', "EXID|이엑스아이디", ""),

    # Ladies' Code
    ("ladies_code__ashley", "Ladies' Code", "Ashley", '"레이디스 코드" 애슐리', "Ladies' Code|Ladies Code|레이디스 코드|레이디스코드", ""),
    ("ladies_code__sojung", "Ladies' Code", "Sojung", '"레이디스 코드" 소정', "Ladies' Code|Ladies Code|레이디스 코드|레이디스코드", ""),
    ("ladies_code__zuny", "Ladies' Code", "Zuny", '"레이디스 코드" 주니', "Ladies' Code|Ladies Code|레이디스 코드|레이디스코드", ""),

    # Lovelyz
    ("lovelyz__baby_soul", "Lovelyz", "Baby Soul", '"러블리즈" 베이비소울', "Lovelyz|러블리즈", ""),
    ("lovelyz__jiae", "Lovelyz", "Jiae", '"러블리즈" 지애', "Lovelyz|러블리즈", ""),
    ("lovelyz__jisoo", "Lovelyz", "Jisoo", '"러블리즈" 지수', "Lovelyz|러블리즈", ""),
    ("lovelyz__mijoo", "Lovelyz", "Mijoo", '"러블리즈" 미주', "Lovelyz|러블리즈", ""),
    ("lovelyz__kei", "Lovelyz", "Kei", '"러블리즈" 케이', "Lovelyz|러블리즈", ""),
    ("lovelyz__jin", "Lovelyz", "Jin", '"러블리즈" 진', "Lovelyz|러블리즈", ""),
    ("lovelyz__sujeong", "Lovelyz", "Sujeong", '"러블리즈" 수정', "Lovelyz|러블리즈", ""),
    ("lovelyz__yein", "Lovelyz", "Yein", '"러블리즈" 예인', "Lovelyz|러블리즈", ""),

    # GFRIEND
    ("gfriend__sowon", "GFRIEND", "Sowon", '"여자친구" 소원', "GFRIEND|여자친구", ""),
    ("gfriend__yerin", "GFRIEND", "Yerin", '"여자친구" 예린', "GFRIEND|여자친구", ""),
    ("gfriend__eunha", "GFRIEND", "Eunha", '"여자친구" 은하', "GFRIEND|여자친구", ""),
    ("gfriend__yuju", "GFRIEND", "Yuju", '"여자친구" 유주', "GFRIEND|여자친구", ""),
    ("gfriend__sinb", "GFRIEND", "SinB", '"여자친구" 신비', "GFRIEND|여자친구", ""),
    ("gfriend__umji", "GFRIEND", "Umji", '"여자친구" 엄지', "GFRIEND|여자친구", ""),

    # BBGIRLS, formerly Brave Girls
    ("bbgirls__minyoung", "BBGIRLS", "Minyoung", '"브브걸" 민영', "BBGIRLS|BB Girls|Brave Girls|브브걸|브레이브걸스", ""),
    ("bbgirls__eunji", "BBGIRLS", "Eunji", '"브브걸" 은지', "BBGIRLS|BB Girls|Brave Girls|브브걸|브레이브걸스", ""),
    ("bbgirls__yuna", "BBGIRLS", "Yuna", '"브브걸" 유나', "BBGIRLS|BB Girls|Brave Girls|브브걸|브레이브걸스", ""),

    # WJSN
    ("wjsn__seola", "WJSN", "Seola", '"우주소녀" 설아', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__bona", "WJSN", "Bona", '"우주소녀" 보나', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__exy", "WJSN", "Exy", '"우주소녀" 엑시', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__soobin", "WJSN", "Soobin", '"우주소녀" 수빈', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__luda", "WJSN", "Luda", '"우주소녀" 루다', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__dawon", "WJSN", "Dawon", '"우주소녀" 다원', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__eunseo", "WJSN", "Eunseo", '"우주소녀" 은서', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__yeoreum", "WJSN", "Yeoreum", '"우주소녀" 여름', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__dayoung", "WJSN", "Dayoung", '"우주소녀" 다영', "WJSN|우주소녀|Cosmic Girls", ""),
    ("wjsn__yeonjung", "WJSN", "Yeonjung", '"우주소녀" 연정', "WJSN|우주소녀|Cosmic Girls", ""),

    # MOMOLAND
    ("momoland__hyebin", "MOMOLAND", "Hyebin", '"모모랜드" 혜빈', "MOMOLAND|모모랜드", ""),
    ("momoland__jane", "MOMOLAND", "Jane", '"모모랜드" 제인', "MOMOLAND|모모랜드", ""),
    ("momoland__nayun", "MOMOLAND", "Nayun", '"모모랜드" 나윤', "MOMOLAND|모모랜드", ""),
    ("momoland__jooe", "MOMOLAND", "JooE", '"모모랜드" 주이', "MOMOLAND|모모랜드", ""),
    ("momoland__ahin", "MOMOLAND", "Ahin", '"모모랜드" 아인', "MOMOLAND|모모랜드", ""),
    ("momoland__nancy", "MOMOLAND", "Nancy", '"모모랜드" 낸시', "MOMOLAND|모모랜드", ""),
]

GROUP_GENDERS: dict[str, str] = {
    "BBGIRLS": "F",
    "DAY6": "M",
    "EXID": "F",
    "GFRIEND": "F",
    "Ladies' Code": "F",
    "Lovelyz": "F",
    "MOMOLAND": "F",
    "N.Flying": "M",
    "VIXX": "M",
    "WJSN": "F",
}


def _load_existing_member_ids() -> set[str]:
    with MEMBERS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("member_id") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("member_id") or "").strip()
        }


def _load_existing_group_genders() -> set[str]:
    if not GROUP_GENDERS_CSV.exists():
        return set()
    with GROUP_GENDERS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("group_name") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("group_name") or "").strip()
        }


def _append_csv_rows(path: Path, rows: list[tuple[str, ...]]) -> None:
    if not rows:
        return
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    with path.open("ab") as handle:
        handle.write(buffer.getvalue().encode("utf-8"))


def main() -> None:
    existing_ids = _load_existing_member_ids()
    member_rows = [row for row in NEW_ROWS if row[0] not in existing_ids]
    _append_csv_rows(MEMBERS_CSV, member_rows)

    existing_groups = _load_existing_group_genders()
    gender_rows = [
        (group_name, gender, "manual_override")
        for group_name, gender in GROUP_GENDERS.items()
        if group_name not in existing_groups
    ]
    _append_csv_rows(GROUP_GENDERS_CSV, gender_rows)

    print(f"Appended members: {len(member_rows)}")
    print(f"Appended group genders: {len(gender_rows)}")
    for member_id, group_name, member_name, *_ in member_rows:
        print(f"  {member_id} ({group_name} - {member_name})")


if __name__ == "__main__":
    main()
