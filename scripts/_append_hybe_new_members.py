"""One-shot helper: append 17 new HYBE members (KATSEYE/aoen/SANTOS BRAVOS) to members.csv.

Mirrors existing CSV style: CRLF line endings, double-quoted search_hint with
embedded literal quotes around the group name.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "members.csv"

# (member_id, group_name, member_name, search_hint, include_terms, exclude_terms)
NEW_ROWS: list[tuple[str, str, str, str, str, str]] = [
    # KATSEYE — HYBE x Geffen
    ("katseye__sophia",   "KATSEYE", "Sophia",   '"KATSEYE" Sophia Laforteza', "KATSEYE|Sophia|소피아", ""),
    ("katseye__lara",     "KATSEYE", "Lara",     '"KATSEYE" Lara Raj',         "KATSEYE|Lara|라라",     ""),
    ("katseye__daniela",  "KATSEYE", "Daniela",  '"KATSEYE" Daniela Avanzini', "KATSEYE|Daniela|다니엘라", ""),
    ("katseye__megan",    "KATSEYE", "Megan",    '"KATSEYE" Megan Skiendiel',  "KATSEYE|Megan|메간",    ""),
    ("katseye__manon",    "KATSEYE", "Manon",    '"KATSEYE" Manon Bannerman',  "KATSEYE|Manon|마논",    ""),

    # aoen — JCONIC (HYBE Japan)
    ("aoen__yuju",     "aoen", "Yuju",     '"aoen" Yuju',     "aoen|Yuju|유주",     "여자친구|GFRIEND|최유나"),
    ("aoen__ruka",     "aoen", "Ruka",     '"aoen" Ruka',     "aoen|Ruka|루카",     ""),
    ("aoen__gaku",     "aoen", "Gaku",     '"aoen" Gaku',     "aoen|Gaku|가쿠",     ""),
    ("aoen__hikaru",   "aoen", "Hikaru",   '"aoen" Hikaru',   "aoen|Hikaru|히카루", "AAA|Kis-My-Ft2"),
    ("aoen__sota",     "aoen", "Sota",     '"aoen" Sota',     "aoen|Sota|소타",     "BE:FIRST"),
    ("aoen__kyosuke",  "aoen", "Kyosuke",  '"aoen" Kyosuke',  "aoen|Kyosuke|쿄스케", ""),
    ("aoen__reo",      "aoen", "Reo",      '"aoen" Reo',      "aoen|Reo|레오",      ""),

    # SANTOS BRAVOS — HYBE Latin America
    ("santos_bravos__alejandro", "SANTOS BRAVOS", "Alejandro", '"Santos Bravos" Alejandro Aramburu', "Santos Bravos|Alejandro Aramburu", ""),
    ("santos_bravos__drew",      "SANTOS BRAVOS", "Drew",      '"Santos Bravos" Drew Venegas',       "Santos Bravos|Drew Venegas",      ""),
    ("santos_bravos__gabi",      "SANTOS BRAVOS", "Gabi",      '"Santos Bravos" Gabi Bermudez',      "Santos Bravos|Gabi Bermudez",     ""),
    ("santos_bravos__kaue",      "SANTOS BRAVOS", "Kaue",      '"Santos Bravos" Kaue Penna',         "Santos Bravos|Kaue Penna",        ""),
    ("santos_bravos__kenneth",   "SANTOS BRAVOS", "Kenneth",   '"Santos Bravos" Kenneth Lavill',     "Santos Bravos|Kenneth Lavill",    ""),
]


def main() -> None:
    existing_ids: set[str] = set()
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if row:
                existing_ids.add(row[0].strip())

    to_add = [r for r in NEW_ROWS if r[0] not in existing_ids]
    skipped = [r[0] for r in NEW_ROWS if r[0] in existing_ids]

    if not to_add:
        print(f"Nothing to add. Skipped (already present): {skipped}")
        return

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    for row in to_add:
        writer.writerow(row)

    with CSV_PATH.open("ab") as f:
        f.write(buf.getvalue().encode("utf-8"))

    print(f"Appended {len(to_add)} rows.")
    if skipped:
        print(f"Skipped (already present): {skipped}")
    print("New member_ids:")
    for r in to_add:
        print(f"  {r[0]}")


if __name__ == "__main__":
    main()
