"""
Repeatedly crawl, extract, curate, and rebuild vectors for members
that don't yet meet the minimum valid-image threshold.

Usage:
    python -m src.replenish
    python -m src.replenish --min-images 5 --max-rounds 5 --limit-per-round 20
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> None:
    cmd = [PYTHON, "-m"] + args
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _valid_counts(embeddings_csv: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not embeddings_csv.exists():
        return counts
    with embeddings_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("is_valid_face") == "true":
                counts[row["member_id"]] += 1
    return counts


def _all_member_ids(members_csv: Path) -> list[str]:
    ids = []
    with members_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mid = row.get("member_id", "").strip()
            if mid:
                ids.append(mid)
    return ids


def _insufficient(members_csv: Path, embeddings_csv: Path, min_images: int) -> list[str]:
    all_ids = _all_member_ids(members_csv)
    counts = _valid_counts(embeddings_csv)
    return [mid for mid in all_ids if counts[mid] < min_images]


def _valid_count_map(embeddings_csv: Path) -> dict[str, int]:
    return dict(_valid_counts(embeddings_csv))


def run(
    members_csv: Path,
    embeddings_csv: Path,
    min_images: int,
    max_rounds: int,
    limit_per_round: int,
    centroid_similarity: float,
    delay_ms: int,
) -> None:
    # 진척이 없는 멤버는 포기 — 매 라운드마다 직전 valid 개수와 비교
    given_up: set[str] = set()
    prev_counts: dict[str, int] = _valid_count_map(embeddings_csv)

    for round_num in range(1, max_rounds + 1):
        lacking = [m for m in _insufficient(members_csv, embeddings_csv, min_images) if m not in given_up]
        if not lacking:
            if given_up:
                print(f"\n남은 멤버는 모두 포기 처리됨. 진척 가능한 멤버는 모두 충족.")
            else:
                print(f"\n모든 멤버가 기준({min_images}장) 충족. 완료.")
            break

        print(f"\n=== Round {round_num}/{max_rounds} ===")
        print(f"기준 미달 멤버 {len(lacking)}명: {', '.join(lacking[:10])}{'...' if len(lacking) > 10 else ''}")
        if given_up:
            print(f"(포기된 멤버 {len(given_up)}명 제외)")

        # 1. 추가 크롤링 (목표 valid 장수만큼만 raw 채움)
        _run([
            "src.crawl",
            "--members", str(members_csv),
            "--limit-per-member", str(min_images),
            "--max-candidates", "150",
            "--delay-ms", str(delay_ms),
            "--member-ids", *lacking,
        ])

        # 2. 얼굴 추출 (신규 이미지만, overwrite 없이)
        # --enable-genderage 로 gender/age 컬럼도 채워야 group_genders.csv 교차검증이 의미있음
        _run([
            "src.extract_faces",
            "--members", str(members_csv),
            "--ctx-id", "0",
            "--enable-genderage",
            "--delete-multi-face",
            "--delete-no-face",
            "--member-ids", *lacking,
        ])

        # 3. 아웃라이어 제거
        if centroid_similarity > 0:
            _run([
                "src.curate_dataset",
                "--min-centroid-similarity", str(centroid_similarity),
                "--keep-invalid",  # 이미 삭제된 row는 건드리지 않음
            ])

        # 4. 멤버 벡터 재빌드
        _run(["src.build_member_vectors"])

        # 결과 확인 + 진척 없는 멤버 포기
        new_counts = _valid_count_map(embeddings_csv)
        newly_given_up: list[str] = []
        for mid in lacking:
            if new_counts.get(mid, 0) <= prev_counts.get(mid, 0):
                given_up.add(mid)
                newly_given_up.append(mid)
        prev_counts = new_counts

        still_lacking = [m for m in _insufficient(members_csv, embeddings_csv, min_images) if m not in given_up]
        print(f"\nRound {round_num} 결과: 기준 미달 {len(still_lacking)}명 남음 (포기 {len(given_up)}명).")
        if newly_given_up:
            print(f"  이번 라운드에 포기된 멤버: {', '.join(newly_given_up[:10])}{'...' if len(newly_given_up) > 10 else ''}")
        if still_lacking:
            print("  남은 미달: " + ", ".join(still_lacking[:10]) + ("..." if len(still_lacking) > 10 else ""))
    else:
        remaining = [m for m in _insufficient(members_csv, embeddings_csv, min_images) if m not in given_up]
        if remaining:
            print(f"\n최대 라운드 도달. 여전히 {len(remaining)}명 기준 미달:")
            for mid in remaining:
                print(f"  {mid}")
    if given_up:
        print(f"\n포기된 멤버 ({len(given_up)}명) — 진척 없음:")
        for mid in sorted(given_up):
            print(f"  {mid}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replenish images for under-represented members.")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--embeddings", default="data/image_embeddings.csv")
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--centroid-similarity", type=float, default=0.3)
    parser.add_argument("--delay-ms", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        members_csv=Path(args.members),
        embeddings_csv=Path(args.embeddings),
        min_images=args.min_images,
        max_rounds=args.max_rounds,
        limit_per_round=args.min_images,
        centroid_similarity=args.centroid_similarity,
        delay_ms=args.delay_ms,
    )


if __name__ == "__main__":
    main()
